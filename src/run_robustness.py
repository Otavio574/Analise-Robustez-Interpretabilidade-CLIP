# run_interpretability.py - VERSÃO OTIMIZADA PARA CLIP
import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from modules.data_handler import load_fgvc_data, load_image
from modules.model_loader import load_clip_model 
from config import DEVICE 

# ============================================================
# TARGET CUSTOMIZADO PARA CLIP (ESSENCIAL!)
# ============================================================

class CLIPOutputTarget:
    """
    Target customizado que calcula a similaridade entre 
    a imagem e o texto da classe alvo (como o CLIP realmente funciona).
    """
    def __init__(self, category, model, text_features):
        self.category = category
        self.model = model
        self.text_features = text_features
        
    def __call__(self, model_output):
        # model_output é o embedding da imagem [batch, dim]
        # Normaliza e calcula similaridade com o texto da classe alvo
        image_features = model_output / model_output.norm(dim=-1, keepdim=True)
        
        # Calcula similaridade cosseno com a classe alvo
        logit_scale = self.model.logit_scale.exp()
        similarity = logit_scale * (image_features @ self.text_features[self.category].unsqueeze(0).T)
        
        return similarity

# ============================================================
# RESHAPE TRANSFORM OTIMIZADO
# ============================================================

def reshape_transform(tensor, height=7, width=7):
    """Transforma tensor do ViT para formato espacial 2D."""
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor[0] if isinstance(tensor, tuple) else tensor
    
    if tensor.numel() == 0:
        raise RuntimeError("Tensor vazio!")
    
    # Corrige formato [Tokens, Batch, Dim] -> [Batch, Tokens, Dim]
    if tensor.dim() == 3 and tensor.size(0) > tensor.size(1) and tensor.size(1) == 1:
        tensor = tensor.transpose(0, 1)
    
    # Remove CLS token
    if tensor.dim() == 3:
        tensor = tensor[:, 1:, :]
    
    # Ajusta dimensões se necessário
    actual_tokens = tensor.size(1)
    expected_tokens = height * width
    
    if actual_tokens != expected_tokens:
        import math
        side = int(math.sqrt(actual_tokens))
        if side * side == actual_tokens:
            height = width = side
    
    # Reshape: [Batch, Tokens, Dim] -> [Batch, Dim, H, W]
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    
    return result

# ============================================================
# CONFIGURAÇÃO
# ============================================================

IMAGE_PATH_TO_ANALYZE = "datasets/FGVC_Aircraft/A340/0658059.jpg" 
TARGET_CLASS_NAME = "A330"  # Classe que o modelo confundiu
CORRECT_CLASS_NAME = "A340"  # Classe correta

# ============================================================
# MÚLTIPLAS VISUALIZAÇÕES
# ============================================================

def generate_multiple_cams(model, preprocess, image_path, target_idx, class_names):
    """Gera múltiplos heatmaps usando diferentes camadas e métodos."""
    
    # Preparar imagem
    rgb_img = load_image(image_path).convert('RGB')
    input_tensor = preprocess(rgb_img).unsqueeze(0).to(DEVICE).float()
    norm_img = np.float32(rgb_img.resize((224, 224))) / 255
    
    # Preparar textos para todas as classes
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(DEVICE)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Diferentes camadas para tentar
    layer_configs = [
        ("last_block", [model.visual.transformer.resblocks[-1]]),
        ("mid_block", [model.visual.transformer.resblocks[len(model.visual.transformer.resblocks)//2]]),
        ("early_block", [model.visual.transformer.resblocks[2]]),
    ]
    
    results = []
    
    for layer_name, target_layers in layer_configs:
        print(f"\n🔍 Testando camada: {layer_name}")
        
        try:
            # GradCAM padrão
            cam = GradCAM(
                model=model.visual, 
                target_layers=target_layers,
                reshape_transform=reshape_transform
            )
            
            # Usa target customizado para CLIP
            targets = [CLIPOutputTarget(target_idx, model, text_features)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Verifica se há ativação significativa
            max_activation = grayscale_cam.max()
            mean_activation = grayscale_cam.mean()
            
            print(f"   ✅ Max ativação: {max_activation:.4f} | Média: {mean_activation:.4f}")
            
            if max_activation > 0.1:  # Só salva se houver ativação significativa
                cam_image = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)
                results.append((layer_name, cam_image, max_activation))
            else:
                print(f"   ⚠️  Ativação muito fraca, pulando...")
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            continue
    
    return results, norm_img

# ============================================================
# VISUALIZAÇÃO COMPARATIVA
# ============================================================

def visualize_comparison(original_img, results, correct_class, target_class):
    """Cria uma visualização com múltiplos heatmaps lado a lado."""
    
    n_results = len(results)
    if n_results == 0:
        print("\n❌ Nenhum heatmap válido foi gerado!")
        return
    
    # Cria figura com subplots
    fig, axes = plt.subplots(1, n_results + 1, figsize=(5 * (n_results + 1), 5))
    
    if n_results == 0:
        axes = [axes]
    
    # Mostra imagem original
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original\n(Real: {correct_class})", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Mostra cada heatmap
    for idx, (layer_name, cam_image, activation) in enumerate(results):
        axes[idx + 1].imshow(cam_image)
        axes[idx + 1].set_title(
            f"{layer_name}\nPredição: {target_class}\n(Max: {activation:.3f})", 
            fontsize=10
        )
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    output_filename = f"gradcam_comparison_{correct_class}_vs_{target_class}.png".replace(' ', '_')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Comparação salva em: {output_filename}")
    
    # Salva também o melhor individualmente
    if results:
        best_result = max(results, key=lambda x: x[2])
        best_cam = Image.fromarray(best_result[1])
        best_filename = f"gradcam_best_{correct_class}_pred_{target_class}.png".replace(' ', '_')
        best_cam.save(best_filename)
        print(f"💾 Melhor heatmap salvo em: {best_filename}")

# ============================================================
# MAIN
# ============================================================

def main_interpretability():
    print("🎯 INICIANDO ANÁLISE DE INTERPRETABILIDADE AVANÇADA (GRAD-CAM)")
    print(f"📍 Dispositivo: {DEVICE}\n")
    
    # Carrega modelo
    model, preprocess = load_clip_model()
    model.eval().to(DEVICE).float()
    
    # Carrega classes
    _, _, class_names = load_fgvc_data()
    class_to_idx = {cls.replace('_', ' '): i for i, cls in enumerate(class_names)}
    target_idx = class_to_idx.get(TARGET_CLASS_NAME)
    
    if target_idx is None:
        print(f"❌ Classe '{TARGET_CLASS_NAME}' não encontrada!")
        return
    
    print(f"🎯 Analisando confusão: {CORRECT_CLASS_NAME} → {TARGET_CLASS_NAME}")
    print(f"   Classe alvo índice: {target_idx}\n")
    
    # Gera múltiplos CAMs
    results, original_img = generate_multiple_cams(
        model, preprocess, IMAGE_PATH_TO_ANALYZE, target_idx, class_names
    )
    
    # Visualiza comparação
    visualize_comparison(original_img, results, CORRECT_CLASS_NAME, TARGET_CLASS_NAME)
    
    print("\n📊 INTERPRETAÇÃO DOS RESULTADOS:")
    print("   • Áreas VERMELHAS/AMARELAS: Alta influência na predição")
    print("   • Áreas VERDES: Influência moderada")
    print("   • Áreas AZUIS: Baixa influência")
    print(f"   • O modelo focou nessas regiões para classificar como '{TARGET_CLASS_NAME}'")
    
    if not results:
        print("\n⚠️  DICA: Se todos os mapas estão fracos, isso pode indicar:")
        print("   1. O modelo CLIP usa mais informação semântica global que features locais")
        print("   2. A decisão está distribuída por toda a imagem")
        print("   3. Tente usar ScoreCAM ou aumentar a resolução da imagem")

if __name__ == '__main__':
    main_interpretability()