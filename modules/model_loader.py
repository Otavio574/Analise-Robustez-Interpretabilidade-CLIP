# modules/model_loader.py

import torch
import clip
# Importa as configurações globais (DEVICE e MODEL_NAME)
from config import DEVICE, MODEL_NAME

def load_clip_model():
    """
    Carrega o modelo CLIP e a função de pré-processamento, 
    movendo o modelo para o dispositivo configurado (CPU ou CUDA).
    """
    print(f"Loading CLIP model '{MODEL_NAME}' on device: {DEVICE}")
    
    # Carrega o modelo e o pré-processador do CLIP
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    
    # Coloca o modelo em modo de avaliação (necessário para inferência)
    model.eval()
    
    return model, preprocess

if __name__ == '__main__':
    # Teste rápido para confirmar que o modelo carrega corretamente
    model, preprocess = load_clip_model()
    print("Modelo CLIP carregado com sucesso!")