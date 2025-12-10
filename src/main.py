import subprocess
import os

# Lista dos scripts na ordem desejada
scripts = [
    "run_first_evaluation.py",
    "run_robustness.py",
    "run_interpretability.py",
    "run_adversarial_attack.py"
]

def main():
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)

        print("=" * 70)
        print(f"▶️ Executando: {script}")
        print("=" * 70)

        # Executa cada script e transmite a saída diretamente no terminal
        result = subprocess.run(
            ["python", script_path],
            text=True
        )

        # Checa por erro
        if result.returncode != 0:
            print(f"❌ Erro ao executar {script}. Interrompendo.")
            break
        else:
            print(f"✅ {script} finalizado com sucesso.\n")

if __name__ == "__main__":
    main()
