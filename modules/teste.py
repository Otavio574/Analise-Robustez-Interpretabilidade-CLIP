import clip
import torch

# Este comando lista os nomes dos modelos que o CLIP pode carregar
print(clip.available_models()) 

# Este comando tenta carregar um modelo real para confirmar o funcionamento
model, preprocess = clip.load("ViT-B/32", device="cpu") 
print("CLIP carregado com sucesso!")