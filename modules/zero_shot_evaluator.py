import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score
from PIL import Image
import traceback

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# ⚠️ AJUSTE ESTE CAMINHO para o diretório que contém as pastas A300/, A310/, etc.
FGVC_DATA_PATH = Path("datasets/fgvc_aircraft") 
RESULTS_DIR = Path("reports_and_results/clip_baseline_2111")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # Ajuste para caber na sua GPU. Se travar, diminua para 16 ou 8.

# ============================================================
# 1. FUNÇÕES DE CARREGAMENTO E DADOS
# ============================================================

def load_image(path):
    """Carrega imagem com PIL e converte para RGB."""
    return Image.open(path).convert("RGB")

def load_fgvc_images_and_labels(data_path: Path):
    """
    Lê os caminhos das imagens e associa cada uma à sua classe (nome da pasta).
    """
    image_paths = []
    class_names_list = []
    
    # Obtém nomes das classes (pastas) e ordena para consistência
    classes = sorted([d.name for d in os.scandir(data_path) if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for class_name in classes:
        class_path = data_path / class_name
        # Busca todas as imagens JPG ou PNG dentro da pasta
        image_paths.extend(class_path.glob("*.jpg"))
        image_paths.extend(class_path.glob("*.png"))
    
    # Cria a lista de rótulos verdadeiros
    labels = []
    for p in image_paths:
        # Extrai o nome da pasta (classe) do caminho
        class_name = p.parent.name 
        class_names_list.append(class_name)
        labels.append(class_to_idx[class_name])

    return image_paths, np.array(labels), classes

# ============================================================
# 2. FUNÇÃO DE EMBEDDING (BASELINE DO SEU TCC)
# ============================================================

def get_text_embedding_baseline(class_name: str, model, clip_library, device):
    """
    Gera embedding de texto usando o template vanilla CLIP ("a photo of a {class} aircraft").
    """
    # Adicionando 'aircraft' para melhor desempenho em classificação fina (Fine-Grained)
    class_readable = class_name.replace('_', ' ')
    text = f"a photo of a {class_readable} aircraft" 
    
    tokens = clip_library.tokenize([text]).to(device)
    
    with torch.no_grad():
        text_embed = model.encode_text(tokens)
        # Normalização
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    
    return text_embed.squeeze(0).cpu()

# ============================================================
# 3. FUNÇÃO DE AVALIAÇÃO (Similaridade e Métricas)
# ============================================================

def evaluate_zero_shot(img_embeds, text_embeds, labels):
    """
    Avaliação zero-shot via similaridade coseno.
    """
    # Calcula similaridade coseno (matriz [N_imgs, N_classes])
    # O tensor 'img_embeds' precisa estar no mesmo device do 'text_embeds' para o cálculo.
    sims = img_embeds.to(text_embeds.device) @ text_embeds.T 
    
    preds = sims.argmax(dim=-1).cpu().numpy()
    
    # Top-1 accuracy
    acc = accuracy_score(labels, preds)
    
    # Top-5 accuracy
    top5_preds = sims.topk(5, dim=-1).indices.cpu().numpy()
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    return acc, top5_acc, preds

# ============================================================
# 4. FUNÇÃO DE ORQUESTRAÇÃO E RELATÓRIO
# ============================================================

def evaluate_zero_shot_fgvc(data_path, model, preprocess, clip_library, device):
    
    # 1. Carregar Dados
    image_paths, labels, class_names = load_fgvc_images_and_labels(data_path)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    print(f"\n   Total imagens: {len(labels)} | Classes: {len(class_names)}")
    
    # 2. Gerar Text Embeddings Baseline
    print("📝 Gerando Text Embeddings Baseline...")
    text_embeds_list = []
    for cls in tqdm(class_names, desc="   Classes"):
        emb = get_text_embedding_baseline(cls, model, clip_library, device) 
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0).to(device) # [N_classes, 512]
    
    # 3. Gerar Image Embeddings (Batch Processing)
    image_embeds_list = []
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="   Gerando Imagem Embeddings"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        
        # Pré-processa e empilha no tensor
        images = [preprocess(load_image(p)).unsqueeze(0) for p in batch_paths]
        image_input = torch.cat(images).to(device)
        
        with torch.no_grad():
            img_embeds = model.encode_image(image_input)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            image_embeds_list.append(img_embeds.cpu()) # Move para CPU após normalização

    image_embeds = torch.cat(image_embeds_list, dim=0).float() # [N_imgs, 512]
    
    # 4. Avaliação Zero-Shot
    acc, top5_acc, preds = evaluate_zero_shot(image_embeds, text_embeds, labels)
    
    # 5. Análise Qualitativa (Exemplos)
    qualitative_results = []
    
    # Pega 2 acertos e 2 erros aleatórios para o relatório
    correct_indices = np.where(labels == preds)[0]
    error_indices = np.where(labels != preds)[0]
    
    # Escolhe no máximo 4 exemplos (2 acertos, 2 erros)
    sample_indices = np.concatenate([
        np.random.choice(correct_indices, min(2, len(correct_indices)), replace=False),
        np.random.choice(error_indices, min(2, len(error_indices)), replace=False)
    ])
    
    for idx in sample_indices:
        pred_class_idx = preds[idx]
        pred_class_name = idx_to_class[pred_class_idx]
        true_class_name = idx_to_class[labels[idx]]
        
        qualitative_results.append({
            "path": str(image_paths[idx]),
            "True Class": true_class_name,
            "Predicted Class": pred_class_name,
            "Result": "CORRETO" if labels[idx] == preds[idx] else "ERROU (Fine-Grained)"
        })
        
    return acc, top5_acc, qualitative_results, len(labels), len(class_names)

def main():
    print("🎯 CLIP Baseline Zero-Shot Evaluation (FGVC Aircraft)")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"🛠️ Path: {FGVC_DATA_PATH}\n")

    if not FGVC_DATA_PATH.is_dir():
        print(f"❌ Erro: O caminho do dataset não existe ou não é um diretório: {FGVC_DATA_PATH}")
        print("Ajuste a variável FGVC_DATA_PATH no topo do script.")
        return

    try:
        print("🔄 Carregando CLIP...")
        # Captura o preprocess necessário para as imagens
        model, preprocess = clip.load(MODEL_NAME, device=DEVICE) 
        model.eval()
        print("✅ Modelo carregado!")

        acc, top5_acc, qual_res, num_imgs, num_classes = evaluate_zero_shot_fgvc(
            FGVC_DATA_PATH, model, preprocess, clip, DEVICE
        )

        # Geração do Relatório JSON
        summary = {
            "dataset": FGVC_DATA_PATH.name,
            "baseline_accuracy_top1": float(acc),
            "baseline_accuracy_top5": float(top5_acc),
            "num_classes": num_classes,
            "num_images": num_imgs,
            "method": "Vanilla CLIP Zero-Shot",
            "template": "a photo of a {class} aircraft",
            "qualitative_examples": qual_res
        }
        
        out_path = RESULTS_DIR / "clip_fgvc_baseline_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        print("\n" + "=" * 70)
        print("📈 RESULTADOS FINAIS (ENTREGA 21/11)")
        print(f"   Top-1 Accuracy: {acc * 100:.2f}%")
        print(f"   Top-5 Accuracy: {top5_acc * 100:.2f}%")
        print(f"   Resultados detalhados salvos em: {out_path}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERRO FATAL DURANTE A EXECUÇÃO: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()