import clip
import torch

# Este comando lista os nomes dos modelos que o CLIP pode carregar
print(clip.available_models()) 
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Este comando tenta carregar um modelo real para confirmar o funcionamento
model, preprocess = clip.load("ViT-B/32", device="cpu") 
print("CLIP carregado com sucesso!")