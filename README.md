# 🔍 Análise de Robustez, Interpretabilidade e Ataques Adversários no CLIP  
Repositório Oficial – Projeto de Tópicos Especiais  
Autor: Otávio Augusto Cavalcanti Neto  
GitHub: https://github.com/Otavio574/Analise-Robustez-Interpretabilidade-CLIP

---

## 📌 Visão Geral

Este repositório contém todo o código utilizado para avaliar o comportamento do modelo **CLIP (ViT-B/32)** em tarefas Fine-Grained, incluindo:

- Avaliação inicial da acurácia Zero-Shot
- Testes de robustez (ruído, transformação e variações de entrada)
- Análise de interpretabilidade (GradCAM em múltiplas camadas)
- Implementação e execução de ataque adversário (PGD)
- Geração de relatórios e heatmaps

O projeto acompanha o relatório técnico desenvolvido na disciplina de **Tópicos Especiais**, cujo objetivo é analisar a *robustez*, a *explicabilidade* e as *vulnerabilidades* de modelos avançados de visão-linguagem.

---

## 🧠 Objetivos do Projeto

1. Aplicar modelos de aprendizagem profunda (CLIP) em uma tarefa de visão computacional Fine-Grained.
2. Avaliar a robustez do modelo sob condições adversas.
3. Investigar o comportamento interno do modelo usando técnicas de interpretabilidade.
4. Desenvolver e aplicar ataques adversários.
5. **Analisar e discutir os insights obtidos sobre o comportamento interno do modelo**, conforme apresentado no Capítulo 7 do relatório.

---

## 📁 Estrutura do Repositório

Analise-Robustez-Interpretabilidade-CLIP/
│
├── src/
│ ├── run_first_evaluation.py # Avaliação Zero-Shot (Top-1 / Top-5)
│ ├── run_robustness.py # Ruído, transformações e variações
│ ├── run_interpretability.py # GradCAM (early, middle, late)
│ ├── run_adversarial_attack.py # PGD Targeted + visualização
│ ├── script_master.py # Executa tudo em ordem automática
│
├── datasets/ # Classe Aircraft (ou link externo)
├── reports_and_results/ # Imagens, heatmaps, gráficos e logs
└── README.md


---

## 📦 Requisitos

- Python 3.8+
- PyTorch + CUDA (opcional, mas recomendado)
- OpenAI CLIP
- torchvision
- pytorch-grad-cam
- numpy, pillow, matplotlib

Instalação rápida:

```bash
pip install -r requirements.txt
```
▶️ Execução Rápida (Modo Automático)

O repositório inclui um script que executa todas as etapas do pipeline automaticamente:
python src/script_master.py

A ordem executada é:

1. run_first_evaluation.py

2. run_robustness.py

3. run_interpretability.py

4. run_adversarial_attack.py

Todos os resultados são armazenados em:

reports_and_results/

▶️ Execução Manual (Etapa por Etapa)

Avaliação Zero-Shot:

python src/run_first_evaluation.py


Robustez:

python src/run_robustness.py


Interpretabilidade (GradCAM):

python src/run_interpretability.py


Ataque Adversário (PGD Targeted):

python src/run_adversarial_attack.py

🔥 Insights Principais (Resumo do Capítulo 7 do Relatório)

Após realizar interpretabilidade, robustez e ataques adversários, os principais insights sobre o comportamento interno do CLIP foram:

1. Foco em features não discriminativas

O modelo ignora características críticas (como o número de motores no A340 vs A330) e se concentra em regiões genéricas como fuselagem, nariz e cauda.

2. Decisões baseadas em semântica global

A camada late domina decisivamente o processo, mostrando que o CLIP se apoia mais em conceitos amplos do que em detalhes estruturais.

3. Arquitetura ViT + treinamento contrastivo limitam Fine-Grained

As descrições textuais do CLIP não carregam granularidade suficiente para distinguir modelos da mesma família.

4. Vulnerabilidade a ataques adversários

Mesmo com perturbação baixa (ε=0.0314), o CLIP foi enganado e classificou A340 como A330 — reforçando sua fragilidade.

5. Implicações práticas

Zero-Shot CLIP não é adequado para FGVC sem ajustes

Fine-tuning supervisionado ou few-shot ajudam

Modelos especializados (TransFG, FFVT) seriam mais apropriados para FGVC-Aircraft

🖼️ Exemplos de Resultados

O relatório inclui imagens como:

- Imagem original vs adversarial

- Perturbação 10×

- Heatmaps GradCAM (early, middle, late)

- Comparações de ativação por camada

As figuras são automaticamente salvas em:

reports_and_results/adversarial_results/

📜 Licença

MIT License

📬 Contato

Se quiser discutir sobre CLIP, interpretabilidade, ataques adversários ou Fine-Grained Vision, só chamar!

