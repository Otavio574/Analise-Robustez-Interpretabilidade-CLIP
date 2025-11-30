# modules/robust_dataset.py

import torch
from torch.utils.data import Dataset
# Importa suas funções existentes
from modules.data_handler import load_image 
from modules.robustness_analyzer import (
    apply_gaussian_noise, apply_rotation
)

class RobustnessDataset(Dataset):
    def __init__(self, image_paths, preprocess, perturbation_type, severity):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.perturbation_type = perturbation_type
        self.severity = severity

    def __len__(self):
        # Retorna o número total de imagens
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Método que carrega, perturba e pré-processa uma única imagem."""
        img_path = self.image_paths[idx]
        img = load_image(img_path)
        
        # 1. Aplica a Perturbação (Lógica executada pelos workers do CPU)
        if self.perturbation_type == "VISUAL_GAUSSIAN_NOISE":
            if self.severity > 0:
                 img = apply_gaussian_noise(img, sigma=self.severity)
        elif self.perturbation_type == "VISUAL_ROTATION":
            img = apply_rotation(img, degrees=int(self.severity))

        # 2. Pré-processa a imagem (Prepara o Tensor para o CLIP)
        img_tensor = self.preprocess(img)
        
        return img_tensor