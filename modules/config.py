# config.py

import torch  # ⬅️ ESSA LINHA RESOLVE O NameError: name 'torch' is not defined
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES DE DISPOSITIVO E PATHS
# ============================================================

# Define o dispositivo de execução (usará a GPU RTX 4060 Ti se a instalação CUDA estiver OK)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Raiz dos Dados (Ajuste se o caminho for diferente)
DATA_ROOT = Path("datasets/FGVC_Aircraft")

# Diretório para salvar os resultados da robustez
RESULTS_DIR = Path("reports_and_results/robustness_analysis_2811")


# ============================================================
# CONFIGURAÇÕES DO MODELO E AVALIAÇÃO
# ============================================================

# Nome do modelo CLIP a ser carregado (ViT-B/32 é o padrão para baseline)
MODEL_NAME = "ViT-B/32"

# Tamanho do batch para processamento na GPU
BATCH_SIZE = 32

# Template padrão para o baseline Zero-Shot
PROMPT_TEMPLATE = "a photo of a {} aircraft"