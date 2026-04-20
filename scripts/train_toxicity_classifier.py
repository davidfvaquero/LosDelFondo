"""
train_toxicity_classifier.py
============================
Fine-tuning de unitary/multilingual-toxic-xlm-roberta para detección de
toxicidad bilingüe (ES/EN) en el contexto de la plataforma DEPORTEData.

El modelo base ya está pre-entrenado para detectar toxicidad multilingüe.
Este script lo adapta (fine-tunea) con ejemplos específicos del dominio deportivo
para mejorar su precisión en el contexto concreto de la aplicación.

Diseñado para ejecutarse en AWS EC2 (CPU o GPU, mucho más ligero que Qwen).
En una instancia t3.medium o g4dn.xlarge tarda 2-5 minutos.

Requisitos:
    pip install transformers datasets scikit-learn torch accelerate

Ejecutar desde la raíz del proyecto:
    python scripts/train_toxicity_classifier.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID      = "unitary/multilingual-toxic-xlm-roberta"
DATASET_PATH  = "data/processed/toxicity_dataset.jsonl"
OUTPUT_DIR    = "models/toxicity-classifier"

# Hiperparámetros — ajustados para el tamaño del dataset (pequeño)
NUM_EPOCHS    = 5
BATCH_SIZE    = 16    # Funciona en CPU o GPU sin problema
LEARNING_RATE = 2e-5
MAX_LENGTH    = 128   # Las frases de chat son cortas, 128 tokens son suficientes
TEST_SIZE     = 0.2   # 20% para evaluación

LABEL2ID = {"non-toxic": 0, "toxic": 1}
ID2LABEL = {0: "non-toxic", 1: "toxic"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> DatasetDict:
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] No se encontró el dataset en: {DATASET_PATH}")
        print("  Ejecuta primero: python scripts/prepare_toxicity_data.py")
        sys.exit(1)

    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    dataset = Dataset.from_list(records)
    split   = dataset.train_test_split(test_size=TEST_SIZE, seed=42)

    n_train   = len(split["train"])
    n_test    = len(split["test"])
    n_toxic   = sum(1 for r in records if r["label"] == 1)
    n_clean   = len(records) - n_toxic

    print(f"[INFO] Dataset cargado: {len(records)} ejemplos totales")
    print(f"[INFO]   No toxicos: {n_clean} | Toxicos: {n_toxic}")
    print(f"[INFO]   Train: {n_train} | Test: {n_test}")

    return split


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOKENIZAR
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_dataset(split: DatasetDict, tokenizer) -> DatasetDict:
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = split.map(tokenize_fn, batched=True)
    # Renombrar 'label' a 'labels' (lo que espera HuggingFace Trainer)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ─────────────────────────────────────────────────────────────────────────────
# 3. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calcular métricas básicas manualmente (sin sklearn dependency en tiempo de eval)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())

    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4a. Tokenizer
    print(f"[INFO] Cargando tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 4b. Dataset
    split     = load_dataset()
    tokenized = tokenize_dataset(split, tokenizer)

    # 4c. Modelo — partimos del modelo ya pre-entrenado en toxicidad
    print(f"[INFO] Cargando modelo: {MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # El modelo base tiene 2 etiquetas, igual que el nuestro
    )

    use_gpu = torch.cuda.is_available()
    print(f"[INFO] GPU disponible: {use_gpu}")

    # 4d. Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_gpu,       # Solo en GPU
        logging_steps=5,
        report_to="none",
        dataloader_pin_memory=use_gpu,
    )

    # 4e. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n[INFO] ─────────────────────────────────────────────────────")
    print("[INFO]  Iniciando fine-tuning del clasificador de toxicidad")
    print("[INFO] ─────────────────────────────────────────────────────\n")
    trainer.train()

    # 4f. Evaluación final completa
    print("\n[INFO] Evaluación final en el conjunto de test:")
    predictions_output = trainer.predict(tokenized["test"])
    preds  = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    print("\n" + classification_report(
        labels, preds,
        target_names=["no-toxica", "toxica"],
        digits=4
    ))
    print("Matriz de confusión:")
    print(confusion_matrix(labels, preds))

    # 4g. Guardar modelo y tokenizer
    print(f"\n[INFO] Guardando modelo en: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 4h. Metadatos
    meta = {
        "base_model":     MODEL_ID,
        "train_examples": len(tokenized["train"]),
        "test_examples":  len(tokenized["test"]),
        "epochs":         NUM_EPOCHS,
        "learning_rate":  LEARNING_RATE,
        "max_length":     MAX_LENGTH,
        "output_dir":     OUTPUT_DIR,
        "labels":         ID2LABEL,
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[✓] Fine-tuning del clasificador de toxicidad completado.")
    print(f"    Modelo guardado en: '{OUTPUT_DIR}'")
    print("    Para usarlo, carga el modelo desde esa ruta con AutoModelForSequenceClassification.")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE INFERENCIA RÁPIDA (para usar desde el dashboard/chatbot)
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(model_dir: str = OUTPUT_DIR):
    """Carga el clasificador fine-tuneado para usar en producción."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def is_toxic(text: str, model, tokenizer, threshold: float = 0.7) -> tuple[bool, float]:
    """
    Devuelve (es_toxico: bool, score_toxicidad: float).
    Usar un threshold de 0.7 para evitar falsos positivos en críticas constructivas.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs       = torch.softmax(logits, dim=-1)[0]
    toxic_score = float(probs[1])
    return toxic_score >= threshold, toxic_score


if __name__ == "__main__":
    main()
