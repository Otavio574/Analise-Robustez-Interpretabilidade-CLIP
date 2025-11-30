# modules/data_handler.py
import os
import glob
from PIL import Image
from config import DATA_ROOT, PROMPT_TEMPLATE
import numpy as np

def load_fgvc_data():
    """
    Carrega caminhos das imagens e converte os rótulos de nome de pasta
    para índices numéricos (0, 1, 2, ...), conforme exigido para o cálculo
    de acurácia.
    """
    all_image_paths = []
    all_label_strings = [] # Guarda os rótulos originais como strings
    
    # 1. Lista todas as classes (nomes das pastas: A300, A310, ...)
    class_names = sorted([d.name for d in os.scandir(DATA_ROOT) if d.is_dir()])
    
    # Dicionário de mapeamento: "Airbus A320" -> 0, "Boeing 747" -> 1
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(DATA_ROOT, class_name)
        
        # Busca todas as imagens JPG ou PNG dentro da pasta
        image_paths = glob.glob(os.path.join(class_path, "*.jpg")) + \
                      glob.glob(os.path.join(class_path, "*.png"))
        
        all_image_paths.extend(image_paths)
        all_label_strings.extend([class_name] * len(image_paths))

    # --- Mapeamento e Conversão (A Correção) ---
    
    # 2. Converte os rótulos de string para índices numéricos
    numerical_labels = np.array([
        class_to_idx[label] 
        for label in all_label_strings
    ], dtype=np.int64) # Garante que o tipo é inteiro
    
    # ---------------------------------------------

    # Retorna os caminhos, os rótulos NUMÉRICOS e a lista de nomes de classes
    return all_image_paths, numerical_labels, class_names

def create_text_prompts(class_names):
    """
    Gera a lista de prompts textuais para o Zero-Shot.
    """
    return [PROMPT_TEMPLATE.format(name) for name in class_names]

def load_image(path):
    """Função wrapper simples para carregar a imagem."""
    return Image.open(path).convert("RGB")