# main.py (Fluxo de Avaliação Zero-Shot)
from modules.data_handler import load_fgvc_data, create_text_prompts, load_image
from modules.model_loader import load_clip_model # Assumindo que você criou este módulo
from modules.zero_shot_evaluator import evaluate_zero_shot # Assumindo que você criou este módulo
import json

def run_initial_evaluation():
    # 1. Preparação
    image_paths, true_labels, class_names = load_fgvc_data()
    text_prompts = create_text_prompts(class_names)
    
    model, processor = load_clip_model()

    # 2. Avaliação (O core da entrega 21/11)
    # A função evaluate_zero_shot fará o loop pelas imagens e o cálculo de similaridade
    accuracy, qualitative_results = evaluate_zero_shot(
        model, 
        processor, 
        image_paths, 
        true_labels, 
        class_names, 
        text_prompts
    )

    # 3. Relatório e Resultados
    results = {
        "application": "CLIP Zero-Shot FGVC Aircraft",
        "baseline_accuracy": accuracy,
        "qualitative_examples": qualitative_results
    }
    
    with open("reports_and_results/results_2111.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Avaliação Inicial Concluída. Acurácia Zero-Shot: {accuracy:.2f}%")

if __name__ == "__main__":
    run_initial_evaluation()