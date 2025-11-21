# config.py
DATA_ROOT = "datasets/FGVC_aircraft" # Ex: apontando para a pasta que contem A300/, A310/, etc.
MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"