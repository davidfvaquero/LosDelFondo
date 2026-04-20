import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel

# Forzar codificación UTF-8 para evitar errores con tildes en Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # En algunas versiones antiguas de Python o entornos específicos
        pass

def test_toxicity_model(models_dir):
    print("\n--- Testing Toxicity Model (Threshold Adjusted to 85%) ---")
    model_path = os.path.join(models_dir, "antiToxicidad", "toxicity-classifier")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval() # Modo evaluación para consistencia
    
    test_texts = [
        "Eres una persona maravillosa, gracias por tu ayuda.",
        "No me gusta cómo trabajas, podrías hacerlo mejor.",
        "Eres un completo inútil y no sirves para nada.",
        "Cállate la boca de una vez, nadie te ha preguntado.",
        "Que tengas un excelente día y muchas gracias por todo."
    ]
    
    threshold = 0.75 # Umbral balanceado (ajustado de 85% para captar insultos claros)
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        # Obtenemos la probabilidad de la clase 'toxic' (índice 1)
        toxic_prob = predictions[0][1].item()
        is_toxic = toxic_prob > threshold
        
        print(f"Text: \"{text}\"")
        print(f"Result: {'TOXIC' if is_toxic else 'NON-TOXIC'} ({toxic_prob:.2%} toxic probability)\n")

def test_qwen_model(models_dir):
    print("\n--- Testing Qwen Finetuned Model (COVID update) ---")
    base_model_path = os.path.join(models_dir, "QwenBase")
    adapter_path = os.path.join(models_dir, "QwenDeporteData", "qwen2.5-finetuned", "checkpoint-1443")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.eval() # Modo evaluación para consistencia
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    questions = [
        "¿Qué porcentaje de la población practica fútbol en España?",
        "¿Cuáles son los beneficios del deporte federado?",
        "Dime un resumen breve del impacto del deporte en la economía española.",
        "¿Cómo afectó el COVID-19 a la práctica deportiva en España según los datos?",
        "¿Cuál es el deporte con mayor número de seguidores oficiales?"
    ]
    
    for q in questions:
        messages = [
            {"role": "system", "content": "Eres un asistente experto en deportes. Responde de forma directa, breve y profesional. Asegúrate de terminar la respuesta con un punto."},
            {"role": "user", "content": q}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print(f"Query: {q}")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=200, 
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
            response_ids = output_ids[0][len(inputs.input_ids[0]):]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"Response: {response}\n")

if __name__ == "__main__":
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    
    try:
        test_toxicity_model(models_dir)
    except Exception as e:
        print(f"Error testing Toxicity model: {e}")
        
    try:
        test_qwen_model(models_dir)
    except Exception as e:
        print(f"Error testing Qwen model: {e}")
