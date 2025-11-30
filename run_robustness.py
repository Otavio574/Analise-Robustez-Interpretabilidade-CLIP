# run_robustness.py (VERSÃO FINAL E OTIMIZADA)
import torch
import clip
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
import os
from modules.robust_dataset import RobustnessDataset # Sua classe de dataset customizada
from torch.utils.data import DataLoader # ⬅️ Importação necessária

# 🚨 IMPORTAÇÕES DOS SEUS MÓDULOS
from modules.data_handler import load_fgvc_data, load_image
from modules.model_loader import load_clip_model 
from modules.robustness_analyzer import (
    apply_gaussian_noise, apply_rotation, introduce_typo, modify_template_semantic
)
from config import DEVICE, BATCH_SIZE

# ============================================================
# CONFIGURAÇÃO E AUXILIARES
# ============================================================
FGVC_DATA_PATH = Path("datasets/fgvc_aircraft/teste") 
RESULTS_DIR = Path("reports_and_results/robustness_analysis_2811")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ AJUSTE SEU BASELINE AQUI
BASELINE_ACC = 0.2499 

# --- Funções Auxiliares ---

def generate_text_embedding(class_names: list, model, clip_library, device, template: str = None):
    """Gera embeddings de texto usando um template, padrão ou modificado."""
    
    text_embeds_list = []
    
    if template is None:
        template = "a photo of a {} aircraft"
        
    for cls in class_names:
        class_readable = cls.replace('_', ' ')
        text = template.format(class_readable)
        
        tokens = clip_library.tokenize([text]).to(device)
        with torch.no_grad():
            text_embed = model.encode_text(tokens)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embeds_list.append(text_embed.squeeze(0).cpu())
            
    return torch.stack(text_embeds_list, dim=0)


def evaluate_zero_shot_core(img_embeds, text_embeds, labels, device):
    """
    Núcleo da avaliação zero-shot (com correções de dtype e acurácia).
    """
    # 1. MOVER PARA O DEVICE E GARANTIR FLOAT32 (Correção do RuntimeError)
    img_embeds = img_embeds.to(device).float()
    text_embeds = text_embeds.to(device).float()
    
    sims = img_embeds @ text_embeds.T  # [N_imgs, N_classes]
    preds = sims.argmax(dim=-1).cpu().numpy()
    
    # 2. GARANTIR QUE LABELS É INTEIRO (Correção do 0.0% de acurácia)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().astype(np.int64)
    elif labels.dtype != np.int64:
        labels = labels.astype(np.int64) 
        
    # Top-1 accuracy
    acc = np.mean(preds == labels)
    
    # Top-5 accuracy
    top5_preds = sims.topk(5, dim=-1).indices.cpu().numpy()
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    return acc, top5_acc


# ============================================================
# FUNÇÃO DE ENCODE DE IMAGEM PERTURBADA
# ============================================================

def generate_perturbed_image_embeddings(image_paths, model, preprocess, device, perturbation_type, severity):
    
    # 1. Cria o Dataset (A etapa que veio antes)
    dataset = RobustnessDataset(image_paths, preprocess, perturbation_type, severity) 
    
    # 2. INSERIR AQUI O BLOCO DO DATALOADER:
    # ----------------------------------------------------------------------
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8, # ⬅️ Otimização para o seu Ryzen 5 5500
        pin_memory=True
    )
    
    image_embeds_list = []
    
    for image_input in tqdm(dataloader, desc=f"   Encode Imagens ({perturbation_type})"):
            
            # O image_input já é o tensor do batch processado (perturbado e pré-processado)
            image_input = image_input.to(device) # Move o tensor PRONTO para a GPU
            
            with torch.no_grad():
                # Codifica na GPU
                img_embeds = model.encode_image(image_input)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                image_embeds_list.append(img_embeds.cpu()) # Traz de volta para a CPU para acumular

        # Concatena todos os batches em uma única matriz de embeddings
    return torch.cat(image_embeds_list, dim=0).float()


# ============================================================
# FUNÇÃO CENTRAL DE AVALIAÇÃO COM PERTURBAÇÃO (MODELO PASSADO)
# ============================================================

def run_perturbed_evaluation(model, preprocess, perturbation_type: str, severity: float = None):
    
    model.eval()
    image_paths, labels, class_names = load_fgvc_data()
    
    # 1. Tratamento Textual
    text_template = None
    if perturbation_type == "TEXTUAL_SEMANTIC":
        text_template = "a picture of a {} plane"
    
    # 2. Geração de Embeddings
    text_embeds = generate_text_embedding(class_names, model, clip, DEVICE, template=text_template)

    if "VISUAL" in perturbation_type:
        # Ataques VISUAIS: gera embeddings perturbados
        image_embeds = generate_perturbed_image_embeddings(
            image_paths, model, preprocess, DEVICE, perturbation_type, severity
        )
    else:
        # Ataques TEXTUAIS: gera embeddings de imagem limpos (reutilizando a função)
        image_embeds = generate_perturbed_image_embeddings(
            image_paths, model, preprocess, DEVICE, "VISUAL_GAUSSIAN_NOISE", severity=0
        )
        
    # 3. Avaliação Zero-Shot
    acc, top5_acc = evaluate_zero_shot_core(image_embeds, text_embeds, labels, DEVICE)
    
    return acc, top5_acc


# ============================================================
# MAIN ORQUESTRAÇÃO (OTIMIZADA PARA CARREGAR MODELO UMA VEZ)
# ============================================================

def main_robustness():
    
    results = {"baseline_top1": BASELINE_ACC, "tests": {}}
    
    print("🎯 INICIANDO ANÁLISE DE ROBUSTEZ (ENTREGA 28/11)")
    
    # 🚨 OTIMIZAÇÃO: Carrega o modelo CLIP APENAS UMA VEZ
    model, preprocess = load_clip_model()
    
    # --- TESTES VISUAIS ---
    print("\n--- TESTES VISUAIS ---")
    # 1. Ruído Gaussiano
    for sigma in [10, 25]:
        acc, _ = run_perturbed_evaluation(model, preprocess, "VISUAL_GAUSSIAN_NOISE", severity=sigma)
        results["tests"][f"Ruído Gaussiano (Sigma={sigma})"] = acc
        
    # 2. Rotação
    for degrees in [5, 15]:
        acc, _ = run_perturbed_evaluation(model, preprocess, "VISUAL_ROTATION", severity=degrees)
        results["tests"][f"Rotação ({degrees} graus)"] = acc

    # --- TESTES TEXTUAIS ---
    print("\n--- TESTES TEXTUAIS ---")
    # 3. Robustez Textual - Semântica
    acc, _ = run_perturbed_evaluation(model, preprocess, "TEXTUAL_SEMANTIC")
    results["tests"]["Semântica ('picture'/'plane')"] = acc
    
    # --- Apresentação dos Resultados ---
    print("\n" + "=" * 70)
    print("📈 RESUMO DA ANÁLISE DE ROBUSTEZ (Top-1)")
    print(f"BASELINE (Top-1): {BASELINE_ACC * 100:.2f}%")
    print("-" * 70)
    
    for test, new_acc in results["tests"].items():
        drop_abs = BASELINE_ACC - new_acc
        drop_perc = (drop_abs / BASELINE_ACC) * 100
        # O print do resultado final agora usará a acurácia correta
        print(f"{test:<35}: {new_acc * 100:.2f}% (Queda: {drop_abs * 100:.2f} p.p. | {drop_perc:.1f}%)")
        
    # Salvar resultados
    out_path = RESULTS_DIR / "robustness_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("-" * 70)
    print(f"Resultados salvos em: {out_path}")
    print("=" * 70)
    
if __name__ == "__main__":
    main_robustness()