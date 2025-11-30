import pandas as pd
# from tabulate import tabulate # Não é necessário se o foco for só salvar em CSV

# ============================================================
# DADOS CONSOLIDADOS DO PROJETO (Manter Intacto)
# ============================================================

# 1. DADOS DA ANÁLISE QUANTITATIVA (BASELINE)
data_quant_baseline = {
    'Métrica': ['Acurácia Top-1 (Baseline)', 'Acurácia Top-5 (Baseline)'],
    'Valor Encontrado': ['24.99%', '65.02%'],
    'Comentário': [
        'Acurácia de partida. Valor baixo é esperado devido à classificação fine-grained Zero-Shot.',
        'Alto valor indica que a classe correta está nas 5 principais predições, confirmando a semântica geral.'
    ]
}
df_quant_baseline = pd.DataFrame(data_quant_baseline)

# 2. DADOS DA ANÁLISE QUANTITATIVA DA ROBUSTEZ
# Os valores foram convertidos de volta para float para facilitar a formatação
data_quant_robustez = {
    'Perturbação Aplicada': [
        'BASELINE (Puro)', 
        'Ruído Gaussiano', 'Ruído Gaussiano', 
        'Rotação Visual', 'Rotação Visual', 
        'Semântica Textual'
    ],
    'Severidade': ['N/A', 'sigma=10', 'sigma=25', '5°', '15°', 'Template modificado'],
    'Acúracía Top-1': [24.99, 24.39, 22.68, 26.22, 23.22, 23.07],
    'Queda Absoluta (p.p.)': ['N/A', 0.60, 2.31, -1.23, 1.77, 1.92],
    'Queda Percentual (%)': ['N/A', 2.4, 9.2, -4.9, 7.1, 7.7]
}
df_quant_robustez = pd.DataFrame(data_quant_robustez)

# 3. DADOS DA ANÁLISE QUALITATIVA (Baseada no seu relatório)
data_qualitativa = {
    'Imagem (Modelo)': ['[Caminho Imagem 1: F-16]', '[Caminho Imagem 2: Falcon 2000]', 
                        '[Caminho Imagem 3: A340]', '[Caminho Imagem 4: Global Express]'],
    'Classe Correta': ['F-16', 'Falcon 2000', 'A340', 'Global Express'],
    'Predição do CLIP': ['F-16', 'Falcon 2000', 'A330', 'Embraer ERJ 145'],
    'Resultado': ['CORRETO', 'CORRETO', 'ERROU (Fine-Grained)', 'ERROU (Fine-Grained)']
}
df_qualitativa = pd.DataFrame(data_qualitativa)


# ============================================================
# SALVAMENTO EM ARQUIVOS CSV
# ============================================================

print("="*80)
print("💾 SALVANDO RESULTADOS EM ARQUIVOS CSV...")

# --- 1. CSV do Baseline ---
csv_baseline_filename = "reports_and_results/" + "matrix_results/" + "relatorio_baseline_quant.csv"
# Salvando a tabela de baseline
df_quant_baseline.to_csv(csv_baseline_filename, index=False, sep=',', encoding='utf-8')
print(f"✅ Baseline Quantitativo salvo em: {csv_baseline_filename}")

# --- 2. CSV da Robustez ---
csv_robustez_filename = "reports_and_results/" + "matrix_results/" + "relatorio_robustez_quant.csv"
# Salvando a tabela de robustez
df_quant_robustez.to_csv(csv_robustez_filename, index=False, sep=',', encoding='utf-8')
print(f"✅ Robustez Quantitativa salva em: {csv_robustez_filename}")

# --- 3. CSV da Análise Qualitativa ---
csv_qualitativa_filename = "reports_and_results/" + "matrix_results/" + "relatorio_analise_qualitativa.csv"
# Salvando a tabela qualitativa
df_qualitativa.to_csv(csv_qualitativa_filename, index=False, sep=',', encoding='utf-8')
print(f"✅ Análise Qualitativa salva em: {csv_qualitativa_filename}")

print("="*80)
print("O script finalizou e os arquivos CSV estão na pasta raiz.")