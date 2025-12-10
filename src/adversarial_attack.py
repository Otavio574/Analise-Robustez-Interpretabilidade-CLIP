# src/adversarial_attack.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Adapta o modelo CLIP para ser atacável (Retorna logits de similaridade)
class AttackedCLIPModel(nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features
        self.logit_scale = self.clip_model.logit_scale.exp()

    def forward(self, image_tensor):
        # A entrada para encode_image do CLIP já deve ser pre-processada
        image_features = self.clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Logits de similaridade
        logits = self.logit_scale * (image_features @ self.text_features.T)
        return logits

def pgd_attack(model, input_image, target_class_idx, text_features, eps=8/255, alpha=2/255, steps=10):
    """Gera um exemplo adversário usando PGD (Targeted Attack)."""
    
    # Adapta o modelo para ter uma interface standard de ataque
    attackable_model = AttackedCLIPModel(model, text_features).to(input_image.device)
    
    # 1. Copia e inicializa perturbação
    x_adv = input_image.clone().detach().to(input_image.device)
    
    # 2. Inicializa ruído aleatório
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1).detach() # Limita no range [0, 1] do tensor
    
    # 3. Iterações PGD
    for _ in range(steps):
        x_adv.requires_grad = True
        
        # 4. Forward Pass e Loss (Targeted: Maximizamos a loss do alvo)
        logits = attackable_model(x_adv)
        
        # Loss: Usamos NLLLoss no log_softmax dos logits
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        # A loss deve ser MINIMIZADA para a classe CORRETA (Untargeted)
        # Ou MAXIMIZADA para a classe ALVO (Targeted)
        # Como estamos fazendo Target, queremos que o logit do alvo seja alto.
        # Loss = -log_prob[target] -> Minimizar isso é Maximizar log_prob[target]
        target_tensor = torch.tensor([target_class_idx] * x_adv.size(0)).to(input_image.device)
        
        # CrossEntropyLoss: minimiza a distância. 
        # Para Target Attack: queremos maximizar a loss *da classe correta*.
        # Para Target Attack: queremos que o modelo classifique como CLASSE_ALVO.
        # Usamos CLASSE_ALVO no targets para o PyTorch calcular a Loss
        loss = nn.CrossEntropyLoss()(logits, target_tensor)
        
        # 5. Backward Pass
        attackable_model.zero_grad()
        loss.backward()
        
        # 6. Atualiza perturbação na direção do gradiente (Targeted: ascendente)
        data_grad = x_adv.grad.data
        x_adv = x_adv.detach() - alpha * torch.sign(data_grad) # Descida na Loss da classe alvo
        
        # 7. Projeção (Limita o ruído e o tensor)
        perturbation = torch.clamp(x_adv - input_image, -eps, eps)
        x_adv = input_image + perturbation
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        
    return x_adv.detach()