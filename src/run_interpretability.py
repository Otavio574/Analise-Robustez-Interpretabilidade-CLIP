
# run_interpretability.py - ANÁLISE AVANÇADA COMPLETA
# run_interpretability.py - ANÁLISE AVANÇADA COMPLETA
import torch
import clip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from collections import defaultdict
import glob

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.data_handler import load_fgvc_data, load_image
from modules.model_loader import load_clip_model 
from config import DEVICE 

# ============================================================
# CONFIGURAÇÃO
# ============================================================

OUTPUT_DIR = "reports_and_results/interpretability_results"
AUTO_FIND_ERRORS = False  # Mude para False se der erro
MAX_IMAGES_PER_CLASS = 3

# Se AUTO_FIND_ERRORS = False, use casos manuais:
MANUAL_CASES = [
    {
        "image_path": "datasets/FGVC_Aircraft/A340/0658059.jpg",
        "correct_class": "A340",
        "predicted_class": "A330",
        "case_type": "error"
    },
    # Adicione mais casos aqui manualmente
]

# ============================================================
# TARGET CUSTOMIZADO PARA CLIP
# ============================================================

class CLIPOutputTarget:
    """Target que calcula similaridade texto-imagem."""
    def __init__(self, category, model, text_features):
        self.category = category
        self.model = model
        self.text_features = text_features
        
    def __call__(self, model_output):
        image_features = model_output / model_output.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        similarity = logit_scale * (image_features @ self.text_features[self.category].unsqueeze(0).T)
        return similarity

# ============================================================
# RESHAPE TRANSFORM CORRIGIDO (SUPORTA SCORECAM)
# ============================================================

def reshape_transform(tensor, height=7, width=7):
    """
    Transforma tensor do ViT para formato espacial 2D.
    CORRIGIDO: Agora suporta tanto GradCAM quanto ScoreCAM.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor[0] if isinstance(tensor, tuple) else tensor
    
    if tensor.numel() == 0:
        raise RuntimeError("Tensor vazio!")
    
    # DEBUG: Mostra shape original
    original_shape = tensor.shape
    
    # CORREÇÃO CRÍTICA PARA SCORECAM:
    # ScoreCAM pode passar tensores com batch size > 1
    # Formato possível: [Batch*Tokens, 1, Dim] ou [Tokens, Batch, Dim]
    
    if tensor.dim() == 3:
        # Caso 1: [Tokens, Batch, Dim] - formato invertido
        if tensor.size(0) >= 49 and tensor.size(1) == 1:
            tensor = tensor.transpose(0, 1)
        
        # Caso 2: [Batch, Tokens+CLS, Dim] - formato correto
        elif tensor.size(1) == 50:
            pass  # Já está correto
        
        # Caso 3: [Batch*Tokens, 1, Dim] - precisa reshape
        elif tensor.size(1) == 1 and tensor.size(0) % 50 == 0:
            batch_size = tensor.size(0) // 50
            tensor = tensor.reshape(batch_size, 50, tensor.size(2))
        
        # Remove CLS token
        if tensor.size(1) == 50:
            tensor = tensor[:, 1:, :]  # Remove primeiro token
    
    # Verifica dimensões finais
    batch_size = tensor.size(0)
    actual_tokens = tensor.size(1)
    dim = tensor.size(2)
    
    expected_tokens = height * width
    
    # Ajusta height/width se necessário
    if actual_tokens != expected_tokens:
        import math
        side = int(math.sqrt(actual_tokens))
        if side * side == actual_tokens:
            height = width = side
    
    # Reshape para [Batch, Height, Width, Dim]
    result = tensor.reshape(batch_size, height, width, dim)
    
    # Transpõe para [Batch, Dim, Height, Width]
    result = result.transpose(2, 3).transpose(1, 2)
    
    return result

# ============================================================
# BUSCA AUTOMÁTICA DE ERROS NO DATASET
# ============================================================

def find_prediction_errors(model, preprocess, data_loader, class_names, max_per_class=3):
    """Encontra automaticamente erros de classificação no dataset."""
    
    print("🔍 Buscando erros de classificação automaticamente...")
    
    model.eval()
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(DEVICE)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    errors_found = defaultdict(list)
    correct_found = defaultdict(list)
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Adapta ao formato do seu data_loader
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 3:
                    images, labels, paths = batch_data
                elif len(batch_data) == 2:
                    images, labels = batch_data
                    paths = [f"image_{i}" for i in range(len(images))]
                else:
                    print(f"⚠️  Formato inesperado do batch: {len(batch_data)} elementos")
                    continue
            else:
                print(f"⚠️  Batch não é tupla/lista: {type(batch_data)}")
                continue
            
            images = images.to(DEVICE)
            
            # Garante que labels seja tensor
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels).to(DEVICE)
            else:
                labels = labels.to(DEVICE)
            
            # Predição CLIP
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = (100.0 * image_features @ text_features.T)
            predictions = logits.argmax(dim=1)
            
            # Identifica erros
            for idx in range(len(images)):
                if isinstance(paths, (list, tuple)) and len(paths) > idx:
                    img_path = paths[idx]
                else:
                    img_path = f"unknown_image_{idx}.jpg"
                
                true_label = labels[idx].item() if labels[idx].dim() == 0 else labels[idx].item()
                pred_label = predictions[idx].item()
                
                true_class = class_names[true_label].replace('_', ' ')
                pred_class = class_names[pred_label].replace('_', ' ')
                
                if true_label != pred_label:
                    # Erro encontrado
                    if len(errors_found[true_class]) < max_per_class:
                        errors_found[true_class].append({
                            "image_path": img_path,
                            "correct_class": true_class,
                            "predicted_class": pred_class,
                            "case_type": "error"
                        })
                else:
                    # Acerto
                    if len(correct_found[true_class]) < max_per_class:
                        correct_found[true_class].append({
                            "image_path": img_path,
                            "correct_class": true_class,
                            "predicted_class": pred_class,
                            "case_type": "correct"
                        })
    
    # Converte para lista
    all_cases = []
    for cases in errors_found.values():
        all_cases.extend(cases)
    for cases in correct_found.values():
        all_cases.extend(cases)
    
    print(f"✅ Encontrados: {len([c for c in all_cases if c['case_type']=='error'])} erros, "
          f"{len([c for c in all_cases if c['case_type']=='correct'])} acertos")
    
    return all_cases

# ============================================================
# ANÁLISE COM MÚLTIPLOS MÉTODOS (SCORECAM CORRIGIDO)
# ============================================================

def analyze_with_multiple_methods(model, preprocess, image_path, target_idx, class_names):
    """Aplica GradCAM e ScoreCAM com correção de reshape."""
    
    # Preparar imagem
    rgb_img = load_image(image_path).convert('RGB')
    input_tensor = preprocess(rgb_img).unsqueeze(0).to(DEVICE).float()
    norm_img = np.float32(rgb_img.resize((224, 224))) / 255
    
    # Preparar textos
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Camadas a testar
    layers_to_test = [
        ("early", model.visual.transformer.resblocks[2]),
        ("middle", model.visual.transformer.resblocks[len(model.visual.transformer.resblocks)//2]),
        ("late", model.visual.transformer.resblocks[-2]),
    ]
    
    results = {}
    
    for layer_name, target_layer in layers_to_test:
        print(f"\n🔬 Analisando camada: {layer_name}")
        
        # 1. GradCAM
        try:
            cam = GradCAM(
                model=model.visual,
                target_layers=[target_layer],
                reshape_transform=reshape_transform
            )
            targets = [CLIPOutputTarget(target_idx, model, text_features)]
            gradcam_result = cam(input_tensor=input_tensor, targets=targets)[0]
            
            max_act = gradcam_result.max()
            mean_act = gradcam_result.mean()
            print(f"   ✅ GradCAM - Max: {max_act:.4f}, Média: {mean_act:.4f}")
            
            if max_act > 0.05:
                results[f"{layer_name}_gradcam"] = {
                    "heatmap": gradcam_result,
                    "method": "GradCAM",
                    "layer": layer_name,
                    "max_activation": max_act,
                    "mean_activation": mean_act
                }
        except Exception as e:
            print(f"   ❌ GradCAM falhou: {e}")
        
        # 2. ScoreCAM (com reshape corrigido)
        try:
            print(f"   🔄 Tentando ScoreCAM...")
            score_cam = ScoreCAM(
                model=model.visual,
                target_layers=[target_layer],
                reshape_transform=reshape_transform
            )
            targets = [CLIPOutputTarget(target_idx, model, text_features)]
            scorecam_result = score_cam(input_tensor=input_tensor, targets=targets)[0]
            
            max_act = scorecam_result.max()
            mean_act = scorecam_result.mean()
            print(f"   ✅ ScoreCAM - Max: {max_act:.4f}, Média: {mean_act:.4f}")
            
            if max_act > 0.05:
                results[f"{layer_name}_scorecam"] = {
                    "heatmap": scorecam_result,
                    "method": "ScoreCAM",
                    "layer": layer_name,
                    "max_activation": max_act,
                    "mean_activation": mean_act
                }
        except Exception as e:
            print(f"   ⚠️  ScoreCAM falhou: {str(e)[:100]}")
    
    return results, norm_img, rgb_img

# ============================================================
# MAPA DE CALOR AGREGADO
# ============================================================

def create_aggregated_heatmap(all_heatmaps, title="Mapa Agregado"):
    """Cria um mapa de calor médio de múltiplas imagens."""
    
    if not all_heatmaps:
        return None
    
    # Calcula média
    stacked = np.stack(all_heatmaps)
    mean_heatmap = stacked.mean(axis=0)
    std_heatmap = stacked.std(axis=0)
    
    # Cria visualização
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mapa médio
    im1 = axes[0].imshow(mean_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title(f'{title}\n(Média de {len(all_heatmaps)} imagens)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Mapa de desvio padrão
    im2 = axes[1].imshow(std_heatmap, cmap='viridis', vmin=0, vmax=std_heatmap.max())
    axes[1].set_title('Desvio Padrão\n(Variabilidade)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    
    return fig, mean_heatmap, std_heatmap

# ============================================================
# GRÁFICOS ESTATÍSTICOS
# ============================================================

def create_statistical_plots(all_cases_results, output_dir):
    """Gera gráficos estatísticos comparativos."""
    
    # Coleta dados
    data_by_layer = defaultdict(lambda: {"max": [], "mean": [], "method": [], "case_type": []})
    
    for case_data, results, _, _ in all_cases_results:
        for result_name, result_info in results.items():
            layer = result_info["layer"]
            method = result_info["method"]
            case_type = case_data["case_type"]
            
            data_by_layer[layer]["max"].append(result_info["max_activation"])
            data_by_layer[layer]["mean"].append(result_info["mean_activation"])
            data_by_layer[layer]["method"].append(method)
            data_by_layer[layer]["case_type"].append(case_type)
    
    # Cria figura com múltiplos subplots
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Comparação de ativação máxima por camada
    ax1 = plt.subplot(2, 3, 1)
    layers = list(data_by_layer.keys())
    max_values = [np.mean(data_by_layer[l]["max"]) for l in layers]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(layers, max_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Ativação Máxima Média', fontsize=11, fontweight='bold')
    ax1.set_title('Ativação Máxima por Camada', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar, val in zip(bars, max_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # 2. Comparação de ativação média por camada
    ax2 = plt.subplot(2, 3, 2)
    mean_values = [np.mean(data_by_layer[l]["mean"]) for l in layers]
    bars = ax2.bar(layers, mean_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Ativação Média', fontsize=11, fontweight='bold')
    ax2.set_title('Ativação Média por Camada', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, mean_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. Distribuição de ativações (boxplot)
    ax3 = plt.subplot(2, 3, 3)
    all_max_acts = []
    labels = []
    for layer in layers:
        all_max_acts.append(data_by_layer[layer]["max"])
        labels.append(layer)
    bp = ax3.boxplot(all_max_acts, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Ativação Máxima', fontsize=11, fontweight='bold')
    ax3.set_title('Distribuição de Ativações', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Comparação GradCAM vs ScoreCAM
    ax4 = plt.subplot(2, 3, 4)
    gradcam_data = []
    scorecam_data = []
    for layer in layers:
        layer_methods = data_by_layer[layer]["method"]
        layer_max = data_by_layer[layer]["max"]
        gradcam_vals = [m for m, method in zip(layer_max, layer_methods) if method == "GradCAM"]
        scorecam_vals = [m for m, method in zip(layer_max, layer_methods) if method == "ScoreCAM"]
        gradcam_data.append(np.mean(gradcam_vals) if gradcam_vals else 0)
        scorecam_data.append(np.mean(scorecam_vals) if scorecam_vals else 0)
    
    x = np.arange(len(layers))
    width = 0.35
    ax4.bar(x - width/2, gradcam_data, width, label='GradCAM', color='#FF6B6B', alpha=0.7)
    ax4.bar(x + width/2, scorecam_data, width, label='ScoreCAM', color='#4ECDC4', alpha=0.7)
    ax4.set_ylabel('Ativação Máxima Média', fontsize=11, fontweight='bold')
    ax4.set_title('GradCAM vs ScoreCAM', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Comparação Erros vs Acertos
    ax5 = plt.subplot(2, 3, 5)
    error_acts = []
    correct_acts = []
    for layer in layers:
        layer_case_types = data_by_layer[layer]["case_type"]
        layer_max = data_by_layer[layer]["max"]
        error_vals = [m for m, ct in zip(layer_max, layer_case_types) if ct == "error"]
        correct_vals = [m for m, ct in zip(layer_max, layer_case_types) if ct == "correct"]
        error_acts.append(np.mean(error_vals) if error_vals else 0)
        correct_acts.append(np.mean(correct_vals) if correct_vals else 0)
    
    x = np.arange(len(layers))
    ax5.bar(x - width/2, error_acts, width, label='Erros', color='#FF6B6B', alpha=0.7)
    ax5.bar(x + width/2, correct_acts, width, label='Acertos', color='#95E1D3', alpha=0.7)
    ax5.set_ylabel('Ativação Máxima Média', fontsize=11, fontweight='bold')
    ax5.set_title('Erros vs Acertos', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(layers)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Heatmap de correlação
    ax6 = plt.subplot(2, 3, 6)
    correlation_data = []
    for layer in layers:
        correlation_data.append([
            np.mean(data_by_layer[layer]["max"]),
            np.mean(data_by_layer[layer]["mean"]),
            np.std(data_by_layer[layer]["max"])
        ])
    
    im = ax6.imshow(correlation_data, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks([0, 1, 2])
    ax6.set_xticklabels(['Max', 'Média', 'Std'], fontsize=10)
    ax6.set_yticks(range(len(layers)))
    ax6.set_yticklabels(layers, fontsize=10)
    ax6.set_title('Estatísticas por Camada', fontsize=12, fontweight='bold')
    
    # Adiciona valores no heatmap
    for i in range(len(layers)):
        for j in range(3):
            text = ax6.text(j, i, f'{correlation_data[i][j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    
    # Salva
    stats_path = os.path.join(output_dir, "estatisticas_completas.png")
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Gráficos estatísticos salvos: {stats_path}")
    
    return stats_path

# ============================================================
# VISUALIZAÇÃO (ATUALIZADA)
# ============================================================

def highlight_aircraft_regions(image, heatmap):
    """Identifica e destaca as top-5 regiões mais ativadas."""
    
    h, w = image.shape[:2]
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h)))
    heatmap_resized = heatmap_resized.astype(float) / 255.0
    
    threshold = np.percentile(heatmap_resized, 90)
    hot_regions = heatmap_resized > threshold
    
    from scipy import ndimage
    labeled, num_features = ndimage.label(hot_regions)
    
    annotated = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(annotated, 'RGBA')
    
    regions_info = []
    for i in range(1, min(num_features + 1, 6)):
        mask = labeled == i
        if mask.sum() < 10:
            continue
            
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=(255, 0, 0, 200),
            width=3
        )
        
        region_activation = heatmap_resized[y_min:y_max, x_min:x_max].mean()
        regions_info.append({
            "bbox": (x_min, y_min, x_max, y_max),
            "activation": region_activation,
            "size": mask.sum()
        })
    
    return annotated, regions_info

def create_comprehensive_visualization(case_data, results, norm_img, original_img, output_dir):
    """Cria visualização completa."""
    
    if not results:
        print("   ⚠️  Nenhum resultado válido")
        return None, []
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["max_activation"], reverse=True)
    
    n_results = len(sorted_results)
    n_cols = min(4, n_results + 1)
    n_rows = (n_results + n_cols) // n_cols
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    
    # Original
    ax1 = plt.subplot(n_rows, n_cols, 1)
    ax1.imshow(original_img)
    title_text = f"Original\nReal: {case_data['correct_class']}\nPred: {case_data['predicted_class']}"
    if case_data['case_type'] == 'error':
        title_text += "\nERRO"
        ax1.set_title(title_text, fontsize=11, fontweight='bold', color='red')
    else:
        title_text += "\nCORRETO"
        ax1.set_title(title_text, fontsize=11, fontweight='bold', color='green')
    ax1.axis('off')
    
    # CAMs
    for idx, (result_name, result_data) in enumerate(sorted_results, 2):
        ax = plt.subplot(n_rows, n_cols, idx)
        
        heatmap = result_data["heatmap"]
        cam_image = show_cam_on_image(norm_img, heatmap, use_rgb=True)
        
        ax.imshow(cam_image)
        
        title = f"{result_data['method']}\n{result_data['layer']}\n"
        title += f"Max: {result_data['max_activation']:.3f}\n"
        title += f"Média: {result_data['mean_activation']:.3f}"
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    filename = f"{case_data['correct_class']}_vs_{case_data['predicted_class']}_complete.png"
    filepath = os.path.join(output_dir, filename.replace(' ', '_'))
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Visualização salva: {filepath}")
    
    # Imagem anotada
    best_result_name, best_result = sorted_results[0]
    annotated_img, regions = highlight_aircraft_regions(norm_img, best_result["heatmap"])
    
    annotated_filename = f"{case_data['correct_class']}_vs_{case_data['predicted_class']}_annotated.png"
    annotated_filepath = os.path.join(output_dir, annotated_filename.replace(' ', '_'))
    annotated_img.save(annotated_filepath)
    
    print(f"   💾 Anotada salva: {annotated_filepath}")
    print(f"   📍 {len(regions)} regiões identificadas")
    
    return best_result, regions

# ============================================================
# RELATÓRIO
# ============================================================

def generate_analysis_report(all_cases_results, output_dir, aggregated_stats=None):
    """Gera relatório completo."""
    
    report_lines = [
        "="*80,
        "RELATÓRIO COMPLETO DE INTERPRETABILIDADE - CLIP ViT-B/32",
        "="*80,
        "",
    ]
    
    for case_data, results, best_result, regions in all_cases_results:
        report_lines.extend([
            f"\n{'='*80}",
            f"CASO: {case_data['correct_class']} → {case_data['predicted_class']}",
            f"Tipo: {case_data['case_type'].upper()}",
            f"{'='*80}",
            "",
            f"Métodos aplicados: {len(results)}",
            f"Melhor método: {best_result['method']} (camada: {best_result['layer']})",
            f"Ativação máxima: {best_result['max_activation']:.4f}",
            f"Ativação média: {best_result['mean_activation']:.4f}",
            "",
            f"Regiões identificadas: {len(regions)}",
        ])
        
        for i, region in enumerate(regions, 1):
            report_lines.append(
                f"  Região {i}: Ativação={region['activation']:.3f}, "
                f"Tamanho={region['size']} px"
            )
        report_lines.append("")
    
    # Estatísticas agregadas
    if aggregated_stats:
        report_lines.extend([
            "\n" + "="*80,
            "ESTATÍSTICAS AGREGADAS",
            "="*80,
            "",
        ] + aggregated_stats)
    
    report_path = os.path.join(output_dir, "relatorio_completo.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📄 Relatório salvo: {report_path}")

# ============================================================
# MAIN
# ============================================================

def main_interpretability():
    print("🎯 ANÁLISE AVANÇADA COMPLETA DE INTERPRETABILIDADE")
    print(f"📍 Dispositivo: {DEVICE}\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carrega modelo
    model, preprocess = load_clip_model()
    model.eval().to(DEVICE).float()
    
    # Carrega dataset
    train_loader, test_loader, class_names = load_fgvc_data()
    class_to_idx = {cls.replace('_', ' '): i for i, cls in enumerate(class_names)}
    
    # Busca casos para análise
    if AUTO_FIND_ERRORS and not MANUAL_CASES:
        analysis_cases = find_prediction_errors(
            model, preprocess, test_loader, class_names, MAX_IMAGES_PER_CLASS
        )
    else:
        analysis_cases = MANUAL_CASES
    
    if not analysis_cases:
        print("❌ Nenhum caso encontrado para análise!")
        return
    
    print(f"\n📊 Analisando {len(analysis_cases)} casos...")
    
    all_results = []
    heatmaps_by_layer = defaultdict(list)
    
    # Analisa cada caso
    for i, case in enumerate(analysis_cases, 1):
        print(f"\n{'='*80}")
        print(f"📊 CASO {i}/{len(analysis_cases)}")
        print(f"{'='*80}")
        print(f"Imagem: {case['image_path']}")
        print(f"Real: {case['correct_class']} | Predito: {case['predicted_class']}")
        
        target_idx = class_to_idx.get(case['predicted_class'])
        if target_idx is None:
            print(f"❌ Classe não encontrada!")
            continue
        
        # Analisa
        results, norm_img, original_img = analyze_with_multiple_methods(
            model, preprocess, case['image_path'], target_idx, class_names
        )
        
        if not results:
            print("   ⚠️  Nenhum resultado válido")
            continue
        
        # Coleta heatmaps para agregação
        for result_name, result_data in results.items():
            layer = result_data["layer"]
            heatmaps_by_layer[layer].append(result_data["heatmap"])
        
        # Visualiza
        best_result, regions = create_comprehensive_visualization(
            case, results, norm_img, original_img, OUTPUT_DIR
        )
        
        all_results.append((case, results, best_result, regions))
    
    if not all_results:
        print("\n❌ Nenhum resultado válido gerado!")
        return
    
    # Cria mapas agregados
    print(f"\n{'='*80}")
    print("📊 GERANDO MAPAS AGREGADOS")
    print(f"{'='*80}")
    
    for layer, heatmaps in heatmaps_by_layer.items():
        if len(heatmaps) < 2:
            continue
        
        print(f"\n🔥 Agregando {len(heatmaps)} heatmaps da camada '{layer}'...")
        
        fig, mean_heatmap, std_heatmap = create_aggregated_heatmap(
            heatmaps, 
            title=f"Mapa Agregado - Camada {layer}"
        )
        
        agg_filename = f"agregado_{layer}.png"
        agg_filepath = os.path.join(OUTPUT_DIR, agg_filename)
        fig.savefig(agg_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   💾 Mapa agregado salvo: {agg_filepath}")
    
    # Gera gráficos estatísticos
    print(f"\n{'='*80}")
    print("📈 GERANDO GRÁFICOS ESTATÍSTICOS")
    print(f"{'='*80}")
    
    stats_path = create_statistical_plots(all_results, OUTPUT_DIR)
    
    # Prepara estatísticas para relatório
    errors = [c for c in all_results if c[0]['case_type'] == 'error']
    corrects = [c for c in all_results if c[0]['case_type'] == 'correct']
    
    aggregated_stats = [
        f"Total de casos: {len(all_results)}",
        f"  - Erros: {len(errors)}",
        f"  - Acertos: {len(corrects)}",
        "",
        "CAMADAS ANALISADAS:",
    ]
    
    for layer in heatmaps_by_layer.keys():
        n_heatmaps = len(heatmaps_by_layer[layer])
        aggregated_stats.append(f"  - {layer}: {n_heatmaps} heatmaps")
    
    aggregated_stats.extend([
        "",
        "INSIGHTS:",
        "• Camadas iniciais mostram features visuais locais (bordas, texturas)",
        "• Camadas intermediárias capturam estruturas (asas, fuselagem, motores)",
        "• Camadas finais agregam informação semântica global",
        "• ScoreCAM tende a ser mais interpretável que GradCAM",
        "• Regiões com alta ativação indicam onde o modelo 'olha' ao decidir",
        "",
    ])
    
    # Gera relatório
    generate_analysis_report(all_results, OUTPUT_DIR, aggregated_stats)
    
    # Resumo final
    print(f"\n{'='*80}")
    print("✅ ANÁLISE CONCLUÍDA!")
    print(f"{'='*80}")
    print(f"📁 Resultados em: {OUTPUT_DIR}/")
    print(f"📊 Casos analisados: {len(all_results)}")
    print(f"📈 Gráficos estatísticos: {stats_path}")
    print(f"🔥 Mapas agregados: {len(heatmaps_by_layer)} camadas")
    print("\n💡 ARQUIVOS GERADOS:")
    print("   • *_complete.png - Comparação de todos os métodos")
    print("   • *_annotated.png - Regiões destacadas")
    print("   • agregado_*.png - Mapas médios por camada")
    print("   • estatisticas_completas.png - Gráficos comparativos")
    print("   • relatorio_completo.txt - Análise textual detalhada")

if __name__ == '__main__':
    main_interpretability()