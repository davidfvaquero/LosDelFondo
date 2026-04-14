from pathlib import Path

import pandas as pd

from scripts.process_data import (
    SOURCE_NAME,
    SOURCE_YEAR,
    build_processed_dataframe,
    generate_source_dataframe,
    persist_datasets,
)


def test_generate_source_dataframe_shape():
    df = generate_source_dataframe()
    assert len(df) == 17
    assert {
        "CCAA",
        "Gasto_Promedio_Hogar_Eur",
        "Licencias_Federadas",
        "Poblacion_Activa_Dep",
    }.issubset(df.columns)


def test_build_processed_dataframe_adds_metadata():
    processed_df = build_processed_dataframe(generate_source_dataframe())
    assert processed_df["anio"].nunique() == 1
    assert processed_df["anio"].iloc[0] == SOURCE_YEAR
    assert processed_df["fuente"].iloc[0] == SOURCE_NAME


def test_persist_datasets_creates_expected_files():
    raw_path, parquet_path = persist_datasets()

    assert Path(raw_path).exists()
    assert Path(parquet_path).exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 17
