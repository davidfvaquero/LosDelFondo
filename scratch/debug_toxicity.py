import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def check():
    model_path = os.path.abspath("models/antiToxicidad/toxicity-classifier")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    texts = [
        "Eres una persona maravillosa, gracias por tu ayuda.",
        "Este es un comentario de prueba muy amable.",
        "Eres genial.",
        "Me gusta tu trabajo."
    ]
    
    print(f"ID to label mapping: {model.config.id2label}")
    
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            print(f"Text: {t}")
            print(f"Logits: {logits}")
            print(f"Probs: {probs}")
            print("-" * 20)

if __name__ == "__main__":
    check()
