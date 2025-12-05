"""
Script para combinar os 3 mapas agregados em uma figura única
para facilitar inserção no relatório.
"""

import matplotlib.pyplot as plt
from PIL import Image
import os

# Diretório dos resultados
RESULTS_DIR = "interpretability_results"

# Carrega as 3 imagens
layers = ['early', 'middle', 'late']
images = []

for layer in layers:
    img_path = os.path.join(RESULTS_DIR, f'agregado_{layer}.png')
    if os.path.exists(img_path):
        images.append(Image.open(img_path))
        print(f"✅ Carregado: agregado_{layer}.png")
    else:
        print(f"❌ Não encontrado: {img_path}")

if len(images) != 3:
    print(f"\n❌ Erro: Esperado 3 imagens, encontrado {len(images)}")
    exit(1)

# Cria figura com 3 colunas
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (ax, img, layer) in enumerate(zip(axes, images, layers)):
    ax.imshow(img)
    ax.set_title(f'{layer.upper()} LAYER', fontsize=16, fontweight='bold')
    ax.axis('off')

plt.tight_layout()

# Salva figura combinada
output_path = os.path.join(RESULTS_DIR, 'figura_5_3_mapas_agregados.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Figura 5.3 combinada salva em: {output_path}")
print("📝 Use esta imagem única no relatório!")