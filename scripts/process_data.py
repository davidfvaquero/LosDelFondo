from __future__ import annotations

import os

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed/deporte_data/anio=2023"
SOURCE_YEAR = 2023
SOURCE_NAME = "MEFD_DEPORTEData"
CCAA_LIST = [
    "Andalucía",
    "Aragón",
    "Asturias, Principado de",
    "Balears, Illes",
    "Canarias",
    "Cantabria",
    "Castilla y León",
    "Castilla - La Mancha",
    "Cataluña",
    "Comunitat Valenciana",
    "Extremadura",
    "Galicia",
    "Madrid, Comunidad de",
    "Murcia, Región de",
    "Navarra, Comunidad Foral de",
    "País Vasco",
    "Rioja, La",
]


def generate_source_dataframe(seed: int = 42) -> pd.DataFrame:
    """Create the reproducible source dataframe used by the app."""
    np.random.seed(seed)
    gasto_base = np.random.normal(300, 50, len(CCAA_LIST))
    licencias_base = gasto_base * np.random.normal(300, 50, len(CCAA_LIST)) + np.random.normal(
        10000, 5000, len(CCAA_LIST)
    )

    return pd.DataFrame(
        {
            "CCAA": CCAA_LIST,
            "Gasto_Promedio_Hogar_Eur": round(pd.Series(gasto_base), 2),
            "Licencias_Federadas": round(pd.Series(licencias_base)).astype(int),
            "Poblacion_Activa_Dep": round(pd.Series(licencias_base) * 1.5).astype(int),
        }
    )


def build_processed_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add the columns expected by the analytical model."""
    df_hechos = df_raw.copy()
    df_hechos["anio"] = SOURCE_YEAR
    df_hechos["fuente"] = SOURCE_NAME
    return df_hechos


def persist_datasets() -> tuple[str, str]:
    """Generate, save and return the raw and parquet dataset paths."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_raw = generate_source_dataframe()
    raw_path = os.path.join(RAW_DIR, "gasto_y_federado_2023.csv")
    df_raw.to_csv(raw_path, index=False)

    df_hechos = build_processed_dataframe(df_raw)
    parquet_path = os.path.join(PROCESSED_DIR, "hechos_indicadores.parquet")
    df_hechos.to_parquet(parquet_path, index=False, engine="pyarrow")

    return raw_path, parquet_path


def main() -> None:
    raw_path, parquet_path = persist_datasets()
    print(f"Datos raw guardados en: {raw_path}")
    print(f"Datos procesados (Parquet) guardados en: {parquet_path}")


if __name__ == "__main__":
    main()
