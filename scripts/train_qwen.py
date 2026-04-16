"""
train_qwen.py
=============
Fine-tuning de Qwen2.5-7B-Instruct con QLoRA usando el dataset generado
por prepare_finetuning_data.py.

Diseñado para ejecutarse en AWS EC2 con GPU (g4dn.xlarge o superior).

Requisitos (instalar en EC2):
    pip install torch transformers trl peft bitsandbytes accelerate datasets

Ejecutar desde la raíz del proyecto:
    python scripts/train_qwen.py
"""

from __future__ import annotations

import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — Ajustar según instancia EC2
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID      = "Qwen/Qwen2.5-7B-Instruct"   # Modelo base (se descarga automáticamente la 1ª vez)
DATASET_PATH  = "data/processed/train_dataset.jsonl"
OUTPUT_DIR    = "models/qwen2.5-7b-deporte"   # Donde se guarda el modelo fine-tuneado

# Hiperparámetros — probados en g4dn.xlarge (T4, 16 GB VRAM)
NUM_EPOCHS      = 3
BATCH_SIZE      = 2      # Reducir a 1 si hay OOM en GPU con menos de 16 GB
GRAD_ACCUM      = 8      # 8 pasos × batch 2 = batch efectivo de 16
LEARNING_RATE   = 2e-4
MAX_SEQ_LEN     = 1024   # Longitud máxima de secuencia
WARMUP_RATIO    = 0.05

# QLoRA — cuantización y LoRA
USE_4BIT        = True   # Poner False solo si no hay GPU (muy lento en CPU)
LORA_R          = 16     # Rango LoRA (más alto = más capacidad, más VRAM)
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

HF_CACHE_DIR    = None   # Cambiar a ruta específica en EC2 si quieres controlar dónde se descarga

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR EL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: str) -> Dataset:
    """Lee el JSONL generado por prepare_finetuning_data.py."""
    if not os.path.exists(path):
        print(f"[ERROR] No se encontró el dataset en: {path}")
        print("  Ejecuta primero: python scripts/prepare_finetuning_data.py")
        sys.exit(1)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[INFO] Dataset cargado: {len(records)} ejemplos.")
    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORMATEAR AL CHAT TEMPLATE DE QWEN
# ─────────────────────────────────────────────────────────────────────────────

def format_example(example: dict, tokenizer) -> dict:
    """Aplica el chat template de Qwen a cada par de mensajes."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFIGURAR Y CARGAR EL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    """Carga el tokenizer y modelo base con cuantización 4-bit (QLoRA)."""
    print(f"[INFO] Cargando tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] GPU disponible: {torch.cuda.is_available()}")

    if USE_4BIT and torch.cuda.is_available():
        print("[INFO] Cargando modelo en 4-bit (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR
        )
    else:
        print("[WARN] Cargando modelo en float32 (CPU o sin cuantización). Muy lento.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            cache_dir=HF_CACHE_DIR
        )

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFIGURAR LORA
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model) -> object:
    """Envuelve el modelo con los adaptadores LoRA."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5a. Dataset
    dataset = load_dataset_from_jsonl(DATASET_PATH)
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"[INFO] Train: {len(split['train'])} | Val: {len(split['test'])} ejemplos")

    # 5b. Modelo y tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 5c. Tokenizar (aplicar chat template antes de pasar a SFTTrainer)
    split = split.map(lambda ex: format_example(ex, tokenizer))

    # 5d. LoRA
    model = apply_lora(model)

    # 5e. TrainingArguments
    use_fp16  = torch.cuda.is_available() and not USE_4BIT
    use_bf16  = torch.cuda.is_available() and USE_4BIT

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",           # Cambiar a "wandb" si usas Weights & Biases
        dataloader_pin_memory=False, # Evitar problemas de memoria en EC2
    )

    # 5f. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    print("\n[INFO] ─────────────────────────────────────")
    print("[INFO]  Iniciando fine-tuning  Qwen2.5-7B")
    print("[INFO] ─────────────────────────────────────\n")
    trainer.train()

    # 5g. Guardar modelo fine-tuneado + tokenizer
    print(f"\n[INFO] Guardando modelo en: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 5h. Guardar metadatos
    meta = {
        "base_model": MODEL_ID,
        "train_examples": len(split["train"]),
        "eval_examples": len(split["test"]),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LEN,
        "output_dir": OUTPUT_DIR
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[✓] Fine-tuning completado con éxito.")
    print(f"    Para usar el modelo, cambia MODEL_ID en data_qa_agent.py a: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
