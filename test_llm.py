import pandas as pd
from dashboard.chatbot import load_models, check_toxicity, generate_llm_response

def main():
    print("Loading models...")
    toxic_clf, llm_pipeline = load_models()
    
    print("Testing toxicity classifier...")
    # Test toxic
    prompt_toxic = "Eres un idiota"
    is_toxic, score = check_toxicity(prompt_toxic, toxic_clf)
    print(f"Toxic prompt: {prompt_toxic} -> {is_toxic} (Score: {score})")
    
    # Test clean
    prompt_clean = "¿Quién gasta más?"
    is_toxic, score = check_toxicity(prompt_clean, toxic_clf)
    print(f"Clean prompt: {prompt_clean} -> {is_toxic} (Score: {score})")
    
    print("Testing LLM generation...")
    # Create fake snippet
    df_fake = pd.DataFrame({
        "CCAA": ["Madrid", "Cataluña"],
        "Gasto Promedio Hogar Eur": [200.5, 190.0],
        "Licencias Federadas": [15000, 18000]
    })
    
    response = generate_llm_response("¿Qué comunidad gasta más, Madrid o Cataluña?", df_fake, llm_pipeline, "ES")
    print("LLM Response:\n", response)

if __name__ == "__main__":
    main()
