"""
ray_finetune.py
===============
Fine-tuning distribuido con Ray Train para el proyecto DEPORTEData.

Entrena en paralelo sobre un clúster de N workers GPU:
  1. Qwen2.5-7B-Instruct  →  QLoRA (SFTTrainer)
  2. unitary/multilingual-toxic-xlm-roberta  →  clasificador de toxicidad

Al terminar, sube ambos modelos a Hugging Face Hub automáticamente.

──────────────────────────────────────────────────────────────────────────────
USO LOCAL (Windows, sin GPU obligatoria — para pruebas):
    python scripts/ray_finetune.py --local --workers 1 --only-toxicity

USO EN CLÚSTER (Linux / EC2 con GPUs):
    # Primero inicia el clúster:
    #   ray start --head --num-cpus=8 --num-gpus=4
    python scripts/ray_finetune.py --workers 4 \\
        --hf-token hf_xxxx --hf-username alfersal

VARIABLES DE ENTORNO equivalentes:
    HF_TOKEN=hf_xxxx
    HF_USERNAME=alfersal
──────────────────────────────────────────────────────────────────────────────
Requisitos (ver requirements-ray.txt):
    pip install "ray[train,data]>=2.10" torch transformers peft trl \\
                bitsandbytes datasets accelerate huggingface_hub scikit-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ─── Logging básico ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ray_finetune")

# ─── Rutas del proyecto ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # raíz del proyecto
DATA_PROCESSED = ROOT / "data" / "processed"
QWEN_DATASET   = DATA_PROCESSED / "train_dataset.jsonl"
TOX_DATASET    = DATA_PROCESSED / "toxicity_dataset.jsonl"
MODELS_DIR     = ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# 0. CLI — Argumentos
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ray distributed fine-tuning — DEPORTEData"
    )
    p.add_argument(
        "--local", action="store_true",
        help="Inicia Ray en modo local (sin cluster). Útil en Windows/pruebas."
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Número de workers Ray Train (default: 4)."
    )
    p.add_argument(
        "--use-gpu", action="store_true", default=False,
        help="Solicitar GPU en cada worker (requiere GPUs disponibles)."
    )
    p.add_argument(
        "--only-toxicity", action="store_true",
        help="Entrenar solo el clasificador de toxicidad (más ligero, sirve sin GPU)."
    )
    p.add_argument(
        "--skip-toxicity", action="store_true",
        help="Saltar el clasificador de toxicidad, entrenar solo Qwen."
    )
    p.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN", ""),
        help="Token de Hugging Face Hub (o variable HF_TOKEN)."
    )
    p.add_argument(
        "--hf-username", default=os.environ.get("HF_USERNAME", "alfersal"),
        help="Usuario/organización en Hugging Face (default: alfersal)."
    )
    p.add_argument(
        "--skip-upload", action="store_true",
        help="No subir modelos a HF Hub (solo entrenar y guardar local)."
    )
    # Hiperparámetros Qwen
    p.add_argument("--qwen-epochs",   type=int,   default=3)
    p.add_argument("--qwen-batch",    type=int,   default=2,
                   help="Batch por GPU. Reducir a 1 si hay OOM.")
    p.add_argument("--qwen-lr",       type=float, default=2e-4)
    p.add_argument("--qwen-max-seq",  type=int,   default=1024)
    p.add_argument("--qwen-lora-r",   type=int,   default=16)
    # Hiperparámetros Toxicidad
    p.add_argument("--tox-epochs",    type=int,   default=5)
    p.add_argument("--tox-batch",     type=int,   default=16)
    p.add_argument("--tox-lr",        type=float, default=2e-5)
    p.add_argument("--tox-max-len",   type=int,   default=128)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 1. INICIALIZACIÓN DE RAY
# ─────────────────────────────────────────────────────────────────────────────

def init_ray(local: bool, num_workers: int, use_gpu: bool) -> None:
    """Conecta al cluster Ray existente o inicia uno local."""
    try:
        import ray
    except ImportError:
        log.error("Ray no está instalado. Ejecuta: pip install 'ray[train,data]'")
        sys.exit(1)

    if local:
        log.info("Iniciando Ray en modo LOCAL (1 nodo, sin cluster externo).")
        # En local: reservamos recursos mínimos para pruebas
        ray.init(
            ignore_reinit_error=True,
            num_cpus=max(num_workers * 2, 4),
            num_gpus=num_workers if use_gpu else 0,
            logging_level=logging.WARNING,
        )
    else:
        log.info("Conectando al cluster Ray (address=auto)...")
        ray.init(address="auto", ignore_reinit_error=True)

    resources = ray.cluster_resources()
    log.info(f"Recursos Ray disponibles: CPUs={resources.get('CPU', 0):.0f}  "
             f"GPUs={resources.get('GPU', 0):.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA DE DATASETS (como Ray Data)
# ─────────────────────────────────────────────────────────────────────────────

def load_ray_dataset(path: Path, name: str):
    """Lee un archivo JSONL y lo convierte en ray.data.Dataset."""
    import ray.data as rd

    if not path.exists():
        log.error(f"Dataset '{name}' no encontrado en: {path}")
        sys.exit(1)

    ds = rd.read_json(str(path))
    log.info(f"Dataset '{name}' cargado: {ds.count()} filas.")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 3. WORKER FUNCTION — QWEN 2.5-7B (QLoRA + SFTTrainer)
# ─────────────────────────────────────────────────────────────────────────────

def train_qwen_worker(config: dict[str, Any]) -> None:
    """
    Función ejecutada en CADA worker Ray.
    Cada worker recibe su shard del dataset y entrena con DDP.
    """
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

    # Ray Train imports
    import ray.train
    from ray.train import get_context

    ctx = get_context()
    rank = ctx.get_local_rank()
    world_size = ctx.get_world_size()

    log.info(f"[Qwen worker {rank}/{world_size}] Iniciando...")

    # ── 3a. Shard del dataset para este worker ──────────────────────────────
    shard = ray.train.get_dataset_shard("train")
    records = list(shard.iter_rows())
    # Extraer el campo 'messages' (estructura JSONL del proyecto)
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # ── 3b. Tokenizer ────────────────────────────────────────────────────────
    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Aplicar chat template
    def format_example(ex: dict) -> dict:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    split = split.map(format_example)

    # ── 3c. Modelo QLoRA ─────────────────────────────────────────────────────
    use_4bit = torch.cuda.is_available() and config.get("use_4bit", True)

    if use_4bit:
        log.info(f"[Qwen worker {rank}] Cargando en 4-bit QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": rank},       # cada worker usa su GPU local
            trust_remote_code=True,
        )
    else:
        log.info(f"[Qwen worker {rank}] Cargando en float32 (CPU/sin GPU)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    # ── 3d. LoRA ─────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_r"] * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    if rank == 0:
        model.print_trainable_parameters()

    # ── 3e. TrainingArguments ────────────────────────────────────────────────
    output_dir = config["output_dir"]
    use_bf16 = torch.cuda.is_available() and use_4bit
    use_fp16 = torch.cuda.is_available() and not use_4bit

    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=8,
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_pin_memory=False,
        # Ray Train requiere no usar sharding interno de HF
        no_cuda=not torch.cuda.is_available(),
    )

    # ── 3f. SFTTrainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=config["max_seq_len"],
        packing=False,
    )

    log.info(f"[Qwen worker {rank}] ─── Iniciando entrenamiento ───")
    trainer.train()

    # ── 3g. Solo el rank 0 guarda el checkpoint final ────────────────────────
    if rank == 0:
        log.info(f"[Qwen worker 0] Guardando modelo en: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        # Guardar metadatos de entrenamiento
        meta = {
            "base_model":     config["model_id"],
            "framework":      "ray_train",
            "num_workers":    world_size,
            "lora_r":         config["lora_r"],
            "epochs":         config["num_epochs"],
            "learning_rate":  config["learning_rate"],
            "max_seq_length": config["max_seq_len"],
            "output_dir":     output_dir,
        }
        with open(os.path.join(output_dir, "training_metadata.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        log.info("[Qwen worker 0] ✓ Entrenamiento Qwen completado.")

    # Reportar métricas al orchestrador Ray
    ray.train.report({"status": "done", "rank": rank})


# ─────────────────────────────────────────────────────────────────────────────
# 4. WORKER FUNCTION — TOXICITY CLASSIFIER (XLM-RoBERTa)
# ─────────────────────────────────────────────────────────────────────────────

def train_toxicity_worker(config: dict[str, Any]) -> None:
    """
    Función ejecutada en CADA worker Ray para el clasificador de toxicidad.
    El dataset es pequeño (~100 ejemplos), por lo que todos los workers
    cargan el dataset completo y usan DDP para el entrenamiento.
    """
    import numpy as np
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    import ray.train
    from ray.train import get_context

    ctx = get_context()
    rank = ctx.get_local_rank()
    world_size = ctx.get_world_size()

    log.info(f"[Toxicity worker {rank}/{world_size}] Iniciando...")

    # ── 4a. Dataset ──────────────────────────────────────────────────────────
    shard = ray.train.get_dataset_shard("train")
    records = list(shard.iter_rows())
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.2, seed=42)

    # Estadísticas (solo rank 0)
    if rank == 0:
        n_toxic = sum(1 for r in records if r.get("label") == 1)
        log.info(f"[Toxicity] Total: {len(records)} | "
                 f"Tóxicos: {n_toxic} | No tóxicos: {len(records) - n_toxic}")

    # ── 4b. Tokenizer ────────────────────────────────────────────────────────
    model_id = config["model_id"]
    max_len  = config["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

    tokenized = DatasetDict({
        "train": split["train"].map(tokenize_fn, batched=True),
        "test":  split["test"].map(tokenize_fn, batched=True),
    })
    tokenized["train"] = tokenized["train"].rename_column("label", "labels")
    tokenized["test"]  = tokenized["test"].rename_column("label", "labels")
    tokenized["train"].set_format("torch",
                                  columns=["input_ids", "attention_mask", "labels"])
    tokenized["test"].set_format("torch",
                                 columns=["input_ids", "attention_mask", "labels"])

    # ── 4c. Métricas ─────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        return {"accuracy": round(acc, 4), "precision": round(prec, 4),
                "recall": round(rec, 4), "f1": round(f1, 4)}

    # ── 4d. Modelo ───────────────────────────────────────────────────────────
    LABEL2ID = {"non-toxic": 0, "toxic": 1}
    ID2LABEL = {0: "non-toxic", 1: "toxic"}

    use_gpu = torch.cuda.is_available()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── 4e. TrainingArguments ────────────────────────────────────────────────
    output_dir = config["output_dir"]
    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_gpu,
        logging_steps=5,
        report_to="none",
        dataloader_pin_memory=use_gpu,
        no_cuda=not use_gpu,
    )

    # ── 4f. Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    log.info(f"[Toxicity worker {rank}] ─── Iniciando entrenamiento ───")
    trainer.train()

    # ── 4g. Rank 0 guarda el modelo ──────────────────────────────────────────
    if rank == 0:
        # Evaluación final y reporte
        pred_output = trainer.predict(tokenized["test"])
        final_metrics = compute_metrics(
            (pred_output.predictions, pred_output.label_ids)
        )
        log.info(f"[Toxicity] Métricas finales: {final_metrics}")

        log.info(f"[Toxicity worker 0] Guardando modelo en: {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        meta = {
            "base_model":     config["model_id"],
            "framework":      "ray_train",
            "num_workers":    world_size,
            "epochs":         config["num_epochs"],
            "learning_rate":  config["learning_rate"],
            "max_length":     config["max_length"],
            "final_metrics":  final_metrics,
            "output_dir":     output_dir,
            "labels":         ID2LABEL,
        }
        with open(os.path.join(output_dir, "training_metadata.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        log.info("[Toxicity worker 0] ✓ Entrenamiento toxicidad completado.")

    ray.train.report({"status": "done", "rank": rank,
                      **{f"final_{k}": v for k, v in
                         (final_metrics if rank == 0 else {}).items()}})


# ─────────────────────────────────────────────────────────────────────────────
# 5. LANZAR ENTRENAMIENTO CON RAY TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_ray_trainer(
    train_fn,
    worker_config: dict,
    datasets: dict,
    num_workers: int,
    use_gpu: bool,
    name: str,
) -> None:
    """Configura y lanza un TorchTrainer de Ray Train."""
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer

    scaling = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={
            "CPU": 2,
            "GPU": 1 if use_gpu else 0,
        },
    )

    run_cfg = RunConfig(
        name=name,
        checkpoint_config=CheckpointConfig(num_to_keep=2),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        train_loop_config=worker_config,
        datasets=datasets,
        scaling_config=scaling,
        run_config=run_cfg,
    )

    log.info(f"\n{'═' * 60}")
    log.info(f"  Iniciando Ray Train: {name}")
    log.info(f"  Workers: {num_workers} | GPU: {use_gpu}")
    log.info(f"{'═' * 60}\n")

    result = trainer.fit()
    log.info(f"[{name}] Resultado: {result.metrics}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. UPLOAD A HUGGING FACE HUB
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_hub(
    model_dir: str,
    repo_id: str,
    hf_token: str,
    model_type: str = "causal_lm",
) -> None:
    """
    Sube el modelo entrenado a Hugging Face Hub.
    Crea el repositorio si no existe.
    """
    try:
        from huggingface_hub import HfApi, login
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError:
        log.error("huggingface_hub no instalado. "
                  "Ejecuta: pip install huggingface_hub")
        return

    if not hf_token:
        log.warning("HF_TOKEN no proporcionado. Saltando subida a Hub.")
        return

    if not Path(model_dir).exists():
        log.error(f"Directorio del modelo no encontrado: {model_dir}")
        return

    log.info(f"\n{'─' * 50}")
    log.info(f"  Subiendo a Hugging Face Hub: {repo_id}")
    log.info(f"{'─' * 50}")

    # Login
    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()

    # Crear repositorio si no existe
    try:
        api.create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=False,
            exist_ok=True,
        )
        log.info(f"Repositorio listo: https://huggingface.co/{repo_id}")
    except Exception as exc:
        log.error(f"Error creando repo {repo_id}: {exc}")
        return

    # Cargar y subir modelo + tokenizer desde el directorio guardado
    try:
        log.info("Cargando modelo para subir...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        if model_type == "causal_lm":
            # Para Qwen: subir los adaptadores LoRA (más ligero que el modelo completo)
            # Intentar cargar como PEFT model, si no como modelo completo
            try:
                from peft import AutoPeftModelForCausalLM
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype="auto",
                )
                log.info("Modelo cargado como PEFT/LoRA.")
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype="auto",
                )
                log.info("Modelo cargado como modelo completo.")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir
            )

        # Generar model card
        _write_model_card(model_dir, repo_id, model_type)

        log.info("Subiendo tokenizer...")
        tokenizer.push_to_hub(repo_id, token=hf_token, private=False)

        log.info("Subiendo modelo (puede tardar varios minutos)...")
        model.push_to_hub(repo_id, token=hf_token, private=False)

        # Subir metadatos de entrenamiento si existen
        meta_path = Path(model_dir) / "training_metadata.json"
        if meta_path.exists():
            api.upload_file(
                path_or_fileobj=str(meta_path),
                path_in_repo="training_metadata.json",
                repo_id=repo_id,
                token=hf_token,
            )
            log.info("Metadatos de entrenamiento subidos.")

        log.info(f"\n✓ Modelo disponible en: https://huggingface.co/{repo_id}")

    except Exception as exc:
        log.error(f"Error subiendo modelo a Hub: {exc}")
        raise


def _write_model_card(model_dir: str, repo_id: str, model_type: str) -> None:
    """Genera un README.md básico como model card."""
    if model_type == "causal_lm":
        base = "Qwen/Qwen2.5-7B-Instruct"
        task = "text-generation"
        desc = (
            "Modelo Qwen2.5-7B fine-tuneado con QLoRA para análisis de "
            "estadísticas deportivas en España. Especializado en responder "
            "preguntas sobre el dataset DEPORTEData (federaciones, gasto, "
            "hábitos deportivos, empleo y empresas del sector)."
        )
    else:
        base = "unitary/multilingual-toxic-xlm-roberta"
        task = "text-classification"
        desc = (
            "Clasificador de toxicidad multilingüe (ES/EN) fine-tuneado "
            "para el contexto deportivo de la plataforma DEPORTEData. "
            "Detecta contenido tóxico en consultas de usuarios."
        )

    card = f"""---
language:
- es
- en
license: apache-2.0
tags:
- sports
- spanish
- deportedata
- fine-tuned
base_model: {base}
pipeline_tag: {task}
---

# {repo_id.split('/')[-1]}

{desc}

## Entrenamiento

- **Modelo base**: `{base}`
- **Framework**: Ray Train + HuggingFace Transformers
- **Clúster**: Ray (4 workers GPU)
- **Proyecto**: [DEPORTEData](https://github.com/alfersal)

## Uso

```python
from transformers import pipeline

pipe = pipeline("{task}", model="{repo_id}")
result = pipe("¿Cuántos deportistas federados hay en España?")
print(result)
```

## Datos de entrenamiento

Dataset propio generado a partir de fuentes oficiales españolas:
Ministerio de Educación, FP y Deporte · INE · Seguridad Social.
"""
    card_path = Path(model_dir) / "README.md"
    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write(card)
    log.info(f"Model card generado: {card_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ACTUALIZAR config.py CON LAS RUTAS DE HF HUB
# ─────────────────────────────────────────────────────────────────────────────

def update_local_config(hf_qwen_repo: str, hf_tox_repo: str) -> None:
    """
    Actualiza config.py local para apuntar a los modelos en HuggingFace Hub,
    permitiendo cargarlos directamente desde el dashboard sin rutas locales.
    """
    config_path = ROOT / "config.py"
    if not config_path.exists():
        log.warning("config.py no encontrado, saltando actualización.")
        return

    content = config_path.read_text(encoding="utf-8")

    # Añadir sección HF Hub si no existe
    hf_section = f"""
# ─── Hugging Face Hub (modelos entrenados con Ray) ─────────────────────────
HF_QWEN_REPO:     str = "{hf_qwen_repo}"
HF_TOXICITY_REPO: str = "{hf_tox_repo}"
# Para usar los modelos de HF Hub en el dashboard, cambia USE_REAL_MODELS = True
# y asegúrate de tener HF_TOKEN en el entorno si los repos son privados.
"""
    if "HF_QWEN_REPO" not in content:
        content += hf_section
        config_path.write_text(content, encoding="utf-8")
        log.info(f"config.py actualizado con repos HF: {hf_qwen_repo}, {hf_tox_repo}")
    else:
        log.info("config.py ya contiene las rutas HF Hub, no se modifica.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Configuración de repositorios HF ────────────────────────────────────
    hf_qwen_repo = f"{args.hf_username}/qwen2.5-7b-deporte"
    hf_tox_repo  = f"{args.hf_username}/toxicity-deporte-es"

    # ── Directorios de salida locales ────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    qwen_output_dir = str(MODELS_DIR / "qwen2.5-7b-deporte")
    tox_output_dir  = str(MODELS_DIR / "toxicity-classifier")

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║      DEPORTEData — Ray Distributed Fine-Tuning          ║")
    log.info(f"║  Workers: {args.workers:<4} | GPU: {str(args.use_gpu):<6}                        ║")
    log.info(f"║  HF User: {args.hf_username:<48} ║")
    log.info("╚══════════════════════════════════════════════════════════╝\n")

    # ── Inicializar Ray ──────────────────────────────────────────────────────
    init_ray(local=args.local, num_workers=args.workers, use_gpu=args.use_gpu)

    # ════════════════════════════════════════════════════════════════════════
    # TAREA A: Fine-tuning Qwen2.5-7B-Instruct
    # ════════════════════════════════════════════════════════════════════════
    if not args.only_toxicity:
        log.info("\n[A] Iniciando fine-tuning Qwen2.5-7B-Instruct...")
        qwen_ds = load_ray_dataset(QWEN_DATASET, "qwen_train")

        qwen_config = {
            "model_id":      "Qwen/Qwen2.5-7B-Instruct",
            "output_dir":    qwen_output_dir,
            "num_epochs":    args.qwen_epochs,
            "batch_size":    args.qwen_batch,
            "learning_rate": args.qwen_lr,
            "max_seq_len":   args.qwen_max_seq,
            "lora_r":        args.qwen_lora_r,
            "use_4bit":      args.use_gpu,   # QLoRA solo si hay GPU
        }

        run_ray_trainer(
            train_fn=train_qwen_worker,
            worker_config=qwen_config,
            datasets={"train": qwen_ds},
            num_workers=args.workers,
            use_gpu=args.use_gpu,
            name="qwen_finetune",
        )

        # Subir a HF Hub
        if not args.skip_upload:
            upload_to_hub(
                model_dir=qwen_output_dir,
                repo_id=hf_qwen_repo,
                hf_token=args.hf_token,
                model_type="causal_lm",
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAREA B: Fine-tuning Toxicity Classifier
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_toxicity:
        log.info("\n[B] Iniciando fine-tuning Toxicity Classifier...")
        tox_ds = load_ray_dataset(TOX_DATASET, "toxicity_train")

        tox_config = {
            "model_id":      "unitary/multilingual-toxic-xlm-roberta",
            "output_dir":    tox_output_dir,
            "num_epochs":    args.tox_epochs,
            "batch_size":    args.tox_batch,
            "learning_rate": args.tox_lr,
            "max_length":    args.tox_max_len,
        }

        run_ray_trainer(
            train_fn=train_toxicity_worker,
            worker_config=tox_config,
            datasets={"train": tox_ds},
            num_workers=args.workers,
            use_gpu=args.use_gpu,
            name="toxicity_finetune",
        )

        # Subir a HF Hub
        if not args.skip_upload:
            upload_to_hub(
                model_dir=tox_output_dir,
                repo_id=hf_tox_repo,
                hf_token=args.hf_token,
                model_type="sequence_classification",
            )

    # ─── Actualizar config.py local ──────────────────────────────────────────
    update_local_config(hf_qwen_repo, hf_tox_repo)

    log.info("\n" + "═" * 60)
    log.info("  ✓ Pipeline de fine-tuning completado con éxito")
    log.info(f"  Qwen  → https://huggingface.co/{hf_qwen_repo}")
    log.info(f"  Toxic → https://huggingface.co/{hf_tox_repo}")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
