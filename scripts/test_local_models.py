import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel

# Forzar codificación UTF-8 robusta para Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul') # Cambia el code page de la terminal a UTF-8

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
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
        "¡Qué bestialidad de partido, ha sido una maldita locura, increíble!",
        "En mi opinión, tu análisis es bastante pobre y carece de rigor deportivo.",
        "Vaya genio estás hecho, seguro que te han dado un premio por tremenda estupidez.",
        "El árbitro estuvo ciego todo el partido, es un sinvergüenza vendido.",
        "No comparto tu visión táctica, creo que el entrenador se equivocó completamente."
    ]
    
    threshold = 0.82 # Umbral muy estricto para evitar falsos positivos con frases como "maravillosa"
    
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
        "Teniendo en cuenta el crecimiento del gasto en deporte hasta 2019, ¿cómo crees que afectó la pandemia de COVID-19 tanto a esos ingresos como al número de federados?",
        "Si observamos la distribución por sexos, ¿qué indican los datos sobre la brecha en el deporte federado y en la práctica general?",
        "¿Es cierto que el tenis es el deporte con más federados y el que más licencias genera en España, muy por encima del fútbol o el baloncesto?",
        "Dime 3 ventajas concretas y basadas en datos de practicar deporte de forma federada frente a la práctica deportiva libre.",
        "Realiza un análisis cruzado: ¿Cómo se relaciona el gasto público por habitante con la evolución del número de clubes deportivos en los últimos 5 años registrados?"
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
