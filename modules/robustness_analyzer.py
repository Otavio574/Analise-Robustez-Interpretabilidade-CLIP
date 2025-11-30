import numpy as np
import torch
import random
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO

# ============================================================
# 1. ROBUSTEZ VISUAL (Perturbações na Imagem)
# ============================================================

def apply_gaussian_noise(image: Image.Image, sigma: float = 25) -> Image.Image:
    """
    Aplica ruído Gaussiano à imagem PIL, simulando baixa qualidade de captura.
    O desvio padrão (sigma) controla a intensidade do ruído.
    """
    img_np = np.array(image, dtype=np.float32)
    
    # Gera ruído gaussiano
    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
    
    # Adiciona ruído e limita os valores entre 0 e 255
    noisy_img_np = img_np + noise
    noisy_img_np = np.clip(noisy_img_np, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img_np)

def apply_blur(image: Image.Image, radius: int = 1) -> Image.Image:
    """
    Aplica desfoque (blur) à imagem, simulando perda de foco.
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

# Para usar o filtro acima, você precisa importar ImageFilter:
# from PIL import Image, ImageFilter 

def apply_rotation(image: Image.Image, degrees: int = 5) -> Image.Image:
    """
    Aplica uma pequena rotação à imagem (simulando desalinhamento).
    O método 'expand=True' ajusta o tamanho da imagem para caber todo o conteúdo rotacionado.
    """
    return image.rotate(degrees, resample=Image.BICUBIC, expand=True)


# ============================================================
# 2. ROBUSTEZ TEXTUAL (Perturbações no Prompt)
# ============================================================

def introduce_typo(text: str) -> str:
    """
    Introduz um erro de digitação simples no prompt de texto.
    (Ex: Troca a letra 'o' por 'i' aleatoriamente).
    """
    if "photo" in text:
        # Troca 'photo' por 'phott'
        return text.replace("photo", "phott")
    elif "aircraft" in text:
        # Troca 'aircraft' por 'aircraftt'
        return text.replace("aircraft", "aircraftt")
    return text

def modify_template_semantic(class_name: str, template: str = "a picture of a {} plane") -> str:
    """
    Altera o template semântico padrão (de 'photo of a {class} aircraft' para 'picture of a {class} plane').
    """
    class_readable = class_name.replace('_', ' ')
    return template.format(class_readable)