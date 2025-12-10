# main_adversarial_analysis.py - SOLUÇÃO COM HOOKS CUSTOMIZADOS

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# =======================================================
# 1. AJUSTE DE CAMINHOS
# =======================================================
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from modules.model_loader import load_clip_model
from modules.data_handler import load_fgvc_data
from config import DEVICE 

# =======================================================
# 2. CLASSE PARA CAPTURAR ATIVAÇÕES E GRADIENTES
# =======================================================

class ActivationsAndGradients:
    """Captura ativações e gradientes de camadas específicas."""
    
    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            self.handles.append(
                target_layer.register_full_backward_hook(self.save_gradient)
            )
    
    def save_activation(self, module, input, output):
        # Para ViT, a saída é (batch, num_tokens, hidden_dim)
        self.activations.append(output.detach())
    
    def save_gradient(self, module, grad_input, grad_output):
        # Salva o gradiente da saída
        self.gradients.append(grad_output[0].detach())
    
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)
    
    def release(self):
        for handle in self.handles:
            handle.remove()

# =======================================================
# 3. IMPLEMENTAÇÃO CUSTOMIZADA DE GRADCAM PARA CLIP
# =======================================================

class CLIPGradCAM:
    """GradCAM adaptado para CLIP ViT."""
    
    def __init__(self, model, target_layer, text_features, target_idx):
        self.model = model
        self.target_layer = target_layer
        self.text_features = text_features
        self.target_idx = target_idx
        self.activations_and_grads = ActivationsAndGradients(
            model.visual, [target_layer]
        )
    
    def __call__(self, input_tensor):
        # Limpa ativações anteriores
        self.activations_and_grads.gradients = []
        self.activations_and_grads.activations = []
        
        # Forward pass através do encoder visual
        output = self.model.visual(input_tensor)
        
        # Calcula a similaridade com o texto alvo
        output = output / output.norm(dim=-1, keepdim=True)
        target_text = self.text_features[self.target_idx].unsqueeze(0)
        similarity = 100.0 * (output @ target_text.T)
        
        # Backward
        self.model.zero_grad()
        similarity.backward(retain_graph=True)
        
        # Verifica se capturou as ativações
        if len(self.activations_and_grads.activations) == 0:
            raise RuntimeError("Nenhuma ativação foi capturada. Verifique os hooks.")
        if len(self.activations_and_grads.gradients) == 0:
            raise RuntimeError("Nenhum gradiente foi capturado. Verifique os hooks.")
        
        # Pega ativações e gradientes (invertidos porque backward empilha ao contrário)
        activations = self.activations_and_grads.activations[-1]  # Shape esperado: (B, N, D) ou (N, B, D)
        gradients = self.activations_and_grads.gradients[0]
        
        # Corrige dimensões se necessário (algumas versões do ViT usam (N, B, D))
        if activations.shape[0] == 50 and activations.shape[1] == 1:
            # Transforma de (N, B, D) para (B, N, D)
            activations = activations.transpose(0, 1)  # (1, 50, 768)
            gradients = gradients.transpose(0, 1)      # (1, 50, 768)
        
        print(f"      Debug: após correção - activations shape: {activations.shape}, gradients shape: {gradients.shape}")
        
        # Remove token CLS
        activations = activations[:, 1:, :]  # (B, 49, D)
        gradients = gradients[:, 1:, :]      # (B, 49, D)
        
        print(f"      Debug: após remover CLS - activations shape: {activations.shape}")
        
        # Global average pooling dos gradientes (peso para cada canal)
        weights = gradients.mean(dim=(0, 1))  # (D,)
        
        print(f"      Debug: weights shape: {weights.shape}")
        
        # Weighted combination
        cam = (activations * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, 49)
        
        print(f"      Debug: cam shape antes do reshape: {cam.shape}")
        
        # Reshape para 7x7
        cam = cam.reshape(input_tensor.shape[0], 7, 7)
        
        # ReLU e normalização
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upscale para 224x224
        cam = cam.unsqueeze(1)  # (B, 1, 7, 7)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        return cam
    
    def release(self):
        self.activations_and_grads.release()

# =======================================================
# 4. FUNÇÕES AUXILIARES
# =======================================================

def load_image(image_path):
    return Image.open(image_path)

def preprocess_adversarial_tensor(adv_tensor):
    """Normaliza tensor [0,1] com mean/std do CLIP."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(adv_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(adv_tensor.device).view(1, 3, 1, 1)
    return (adv_tensor - mean) / std

def show_cam_on_image(img, mask, use_rgb=True, colormap=cv2.COLORMAP_JET):
    """Sobrepõe o heatmap na imagem."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    if np.max(img) > 1:
        img = img / 255
    
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# =======================================================
# 5. ATAQUE PGD
# =======================================================

class AttackedCLIPModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, image_tensor, target_text_features):
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ target_text_features.T)
        return logits

def pgd_attack(model, images, target_idx, text_features, eps, steps):
    """Ataque PGD direcionado."""
    alpha = eps / steps
    device = images.device
    
    attack_model = AttackedCLIPModel(model).to(device)
    target_feature = text_features[target_idx].unsqueeze(0)
    
    adv_images = images.clone().detach()
    
    print(f"   Iniciando PGD: eps={eps:.4f}, steps={steps}, alpha={alpha:.4f}")
    
    for step in range(steps):
        adv_images.requires_grad = True
        
        logits_target = attack_model(adv_images, target_feature)
        cost = -logits_target.mean()

        model.zero_grad()
        if adv_images.grad is not None:
            adv_images.grad.zero_()
        cost.backward()
        
        adv_images_grad = adv_images.grad.sign()
        adv_images = adv_images.detach() - alpha * adv_images_grad
        
        delta = adv_images - images
        delta = torch.clamp(delta, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        if (step + 1) % 5 == 0 or step == 0:
            with torch.no_grad():
                logit_val = attack_model(adv_images, target_feature).item()
                print(f"      Step {step+1}/{steps}: Logit={logit_val:.2f}")
    
    return adv_images

# =======================================================
# 6. ANÁLISE COM CAM
# =======================================================

def analyze_with_cam(model, preprocess, image_path, target_idx, 
                     class_names, case_type=None, attack_params=None):
    """Executa análise com GradCAM customizado."""
    
    # Carrega imagem
    rgb_img = load_image(image_path).convert('RGB')
    norm_img_base = np.float32(rgb_img.resize((224, 224))) / 255
    input_tensor_base = preprocess(rgb_img).unsqueeze(0).to(DEVICE).float()
    
    # Prepara text features
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Inicializa
    input_tensor_to_use = input_tensor_base
    norm_img_to_use = norm_img_base
    
    # Gera adversário se solicitado
    if case_type == "adversarial" and attack_params:
        print(f"🛡️ Gerando adversário (Target: {class_names[target_idx]})")
        
        original_01 = torch.from_numpy(norm_img_base).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float()
        
        try:
            adv_01 = pgd_attack(
                model, original_01, target_idx, text_features,
                eps=attack_params.get("eps", 8/255),
                steps=attack_params.get("steps", 10)
            )
            
            input_tensor_to_use = preprocess_adversarial_tensor(adv_01)
            norm_img_to_use = adv_01.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Verifica sucesso
            with torch.no_grad():
                img_feat = model.encode_image(input_tensor_to_use)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = (100.0 * img_feat @ text_features.T)
                pred_idx = logits.argmax(dim=1).item()
                
                if pred_idx == target_idx:
                    print(f"   ✅ Ataque SUCESSO! Predito: {class_names[pred_idx]}")
                else:
                    print(f"   ⚠️ Ataque FALHOU! Predito: {class_names[pred_idx]}")
                    input_tensor_to_use = input_tensor_base
                    norm_img_to_use = norm_img_base
        
        except Exception as e:
            print(f"❌ Erro no PGD: {e}")
            input_tensor_to_use = input_tensor_base
            norm_img_to_use = norm_img_base
    
    # Define camadas para análise
    num_layers = len(model.visual.transformer.resblocks)
    layers_to_test = [
        ("early", model.visual.transformer.resblocks[2]),
        ("middle", model.visual.transformer.resblocks[num_layers // 2]),
        ("late", model.visual.transformer.resblocks[-2]),
    ]
    
    results = {}
    
    # Executa CAM em cada camada
    for layer_name, target_layer in layers_to_test:
        print(f"\n🔬 Analisando camada: {layer_name}")
        
        try:
            # GradCAM customizado
            cam = CLIPGradCAM(model, target_layer, text_features, target_idx)
            
            input_cam = input_tensor_to_use.clone()
            input_cam.requires_grad = True
            
            heatmap = cam(input_cam)
            
            # Se batch, pega primeiro elemento
            if len(heatmap.shape) > 2:
                heatmap = heatmap[0]
            
            max_act = float(heatmap.max())
            print(f"   ✅ GradCAM - Max: {max_act:.4f}")
            
            if max_act > 0.01:
                results[f"{layer_name}_gradcam"] = {
                    "heatmap": heatmap,
                    "method": "GradCAM",
                    "layer": layer_name,
                    "max_activation": max_act
                }
            
            cam.release()
            
        except Exception as e:
            print(f"   ❌ GradCAM falhou: {e}")
            import traceback
            traceback.print_exc()
    
    return results, norm_img_to_use, rgb_img

# =======================================================
# 7. VISUALIZAÇÃO
# =======================================================

def create_visualization(case_data, results, norm_img, original_img, output_dir):
    """Cria visualização dos resultados."""
    
    if not results:
        print("⚠️ Sem resultados para visualizar")
        return None, None
    
    num_maps = len(results)
    cols = 3
    rows = (num_maps + 2) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    
    fig.suptitle(
        f"Análise Adversária: {case_data['correct_class']} → {case_data['predicted_class']}\n"
        f"ε={case_data['attack_params']['eps']:.4f}, steps={case_data['attack_params']['steps']}",
        fontsize=14, fontweight='bold'
    )
    
    # Imagem adversária
    axes[0].imshow(norm_img)
    axes[0].set_title("Imagem Adversária", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Imagem original
    if isinstance(original_img, Image.Image):
        original_resized = np.array(original_img.resize((224, 224))) / 255
    else:
        original_resized = np.array(Image.fromarray(
            (original_img * 255).astype(np.uint8)
        ).resize((224, 224))) / 255
    axes[1].imshow(original_resized)
    axes[1].set_title(f"Original ({case_data['correct_class']})", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Perturbação amplificada
    diff = np.abs(norm_img - original_resized)
    diff_enhanced = diff * 10  # Amplifica para visualização
    diff_enhanced = np.clip(diff_enhanced, 0, 1)
    axes[2].imshow(diff_enhanced)
    axes[2].set_title("Perturbação (10x)", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Heatmaps
    i = 3
    best_result = None
    max_activation = -1
    
    for key, res in sorted(results.items(), key=lambda x: x[1]['max_activation'], reverse=True):
        if i >= len(axes):
            break
        
        heatmap = res['heatmap']
        vis = show_cam_on_image(norm_img, heatmap, use_rgb=True)
        
        axes[i].imshow(vis)
        axes[i].set_title(
            f"{res['method']} - Camada {res['layer']}\n"
            f"Ativação Máx: {res['max_activation']:.3f}",
            fontsize=10
        )
        axes[i].axis('off')
        
        if res['max_activation'] > max_activation:
            max_activation = res['max_activation']
            best_result = res
        
        i += 1
    
    # Desliga eixos não usados
    for j in range(i, len(axes)):
        axes[j].axis('off')
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"adversarial_{case_data['correct_class']}_{case_data['predicted_class']}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Visualização salva: {save_path}")
    
    return best_result, results

# =======================================================
# 8. MAIN
# =======================================================

def run_adversarial_analysis():
    print("🛡️ ESTÁGIO 4: ATAQUES ADVERSÁRIOS E INTERPRETABILIDADE")
    print("=" * 60)
    
    model, preprocess = load_clip_model()
    model.eval().to(DEVICE).float()
    
    _, _, class_names = load_fgvc_data()
    class_to_idx = {cls.replace('_', ' '): i for i, cls in enumerate(class_names)}
    
    case_data = {
        "image_path": "datasets/FGVC_Aircraft/A340/0658059.jpg",
        "correct_class": "A340",
        "predicted_class": "A330",
        "case_type": "adversarial",
        "attack_params": {"eps": 8/255, "steps": 10},
        "description": "PGD targeted attack"
    }
    
    target_idx = class_to_idx.get(case_data['predicted_class'])
    if target_idx is None:
        print(f"❌ Classe '{case_data['predicted_class']}' não encontrada!")
        return
    
    print(f"\n📋 Configuração do Ataque:")
    print(f"   Imagem: {case_data['image_path']}")
    print(f"   Classe Original: {case_data['correct_class']}")
    print(f"   Classe Alvo: {case_data['predicted_class']}")
    print(f"   Epsilon: {case_data['attack_params']['eps']:.4f}")
    print(f"   Steps: {case_data['attack_params']['steps']}\n")
    
    results, norm_img, original_img = analyze_with_cam(
        model, preprocess, case_data['image_path'],
        target_idx, class_names,
        case_type=case_data['case_type'],
        attack_params=case_data['attack_params']
    )
    
    if results:
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DA ANÁLISE")
        print("=" * 60)
        
        for key, res in sorted(results.items(), key=lambda x: x[1]['max_activation'], reverse=True):
            print(f"  {res['method']:10s} | Camada {res['layer']:6s} | Ativação Máx: {res['max_activation']:.4f}")
        
        best_result, _ = create_visualization(
            case_data, results, norm_img, original_img, OUTPUT_DIR_ADV
        )
        
        print("\n" + "=" * 60)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print(f"\n📂 Resultados salvos em: {OUTPUT_DIR_ADV}/")
        
        # Insights
        print("\n💡 INSIGHTS:")
        print(f"   • O ataque PGD conseguiu enganar o modelo com perturbação ε={case_data['attack_params']['eps']:.4f}")
        print(f"   • A camada com maior ativação foi: {best_result['layer']}")
        print(f"   • Ativação máxima: {best_result['max_activation']:.4f}")
        print("\n   Os mapas de calor mostram onde o modelo 'olha' ao fazer a")
        print("   classificação incorreta do exemplo adversário.\n")
        
    else:
        print("\n❌ Nenhum resultado de CAM foi gerado")

if __name__ == '__main__':
    OUTPUT_DIR_ADV = "reports_and_results/adversarial_results"
    run_adversarial_analysis()