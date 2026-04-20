import os
from huggingface_hub import snapshot_download

def download_models():
    # Define local models directory
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Download Toxicity Classifier
    print("Downloading Toxicity Classifier (alfersal04/antiToxicidad)...")
    toxicity_path = os.path.join(models_dir, "antiToxicidad")
    snapshot_download(
        repo_id="alfersal04/antiToxicidad",
        local_dir=toxicity_path,
        allow_patterns=["toxicity-classifier/*"]
    )
    print(f"Toxicity Classifier downloaded to: {toxicity_path}")
    
    # 2. Download Qwen Finetuned Adapter
    print("Downloading Qwen Finetuned Adapter (alfersal04/QwenDeporteData)...")
    qwen_adapter_path = os.path.join(models_dir, "QwenDeporteData")
    snapshot_download(
        repo_id="alfersal04/QwenDeporteData",
        local_dir=qwen_adapter_path,
        allow_patterns=["qwen2.5-finetuned/checkpoint-1443/*"]
    )
    print(f"Qwen Adapter downloaded to: {qwen_adapter_path}")
    
    # 3. Download Qwen Base Model
    print("Downloading Qwen Base Model (Qwen/Qwen2.5-1.5B-Instruct)...")
    qwen_base_path = os.path.join(models_dir, "QwenBase")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        local_dir=qwen_base_path
    )
    print(f"Qwen Base Model downloaded to: {qwen_base_path}")

if __name__ == "__main__":
    download_models()
