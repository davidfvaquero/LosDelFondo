from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_YEAR = 2023
SOURCE_NAME = "gasto_y_federado_2023"

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT_DIR / "data" / "raw" / f"{SOURCE_NAME}.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "deporte_data" / f"anio={SOURCE_YEAR}"
PROCESSED_PARQUET = PROCESSED_DIR / "hechos_indicadores.parquet"


def generate_source_dataframe() -> pd.DataFrame:
    """Carga el CSV fuente consolidado usado por la app y los tests."""
    return pd.read_csv(SOURCE_CSV)


def build_processed_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    """Añade metadatos de partición al dataset base."""
    processed_df = source_df.copy()
    processed_df["anio"] = SOURCE_YEAR
    processed_df["fuente"] = SOURCE_NAME
    return processed_df


def persist_datasets() -> tuple[str, str]:
    """Genera el parquet procesado y devuelve las rutas del origen y del parquet."""
    source_df = generate_source_dataframe()
    processed_df = build_processed_dataframe(source_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(PROCESSED_PARQUET, index=False)

    return str(SOURCE_CSV), str(PROCESSED_PARQUET)


def main() -> None:
    source_path, parquet_path = persist_datasets()
    print(f"Fuente cargada: {source_path}")
    print(f"Parquet generado: {parquet_path}")


if __name__ == "__main__":
    main()
