from pathlib import Path

import pandas as pd

from scripts.process_data import (
    FEDERADOS_PARQUET,
    GASTO_PARQUET,
    PARTITIONED_ROOT,
    SOURCE_NAME,
    SOURCE_YEAR,
    build_federados_dataframe,
    build_gasto_dataframe,
    build_partitioned_dataframe,
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


def test_build_partitioned_dataframe_has_expected_granularity():
    processed_df = build_processed_dataframe(generate_source_dataframe())
    partitioned_df = build_partitioned_dataframe(processed_df)
    assert len(partitioned_df) == 17 * 3
    assert {"anio", "ccaa", "indicador", "valor"}.issubset(partitioned_df.columns)


def test_build_raw_dataframes_include_clean_partition_columns():
    federados_df = build_federados_dataframe()
    gasto_df = build_gasto_dataframe()

    assert {"periodo", "ccaa_limpia", "Total_Num", "archivo_origen"}.issubset(federados_df.columns)
    assert {"periodo", "ccaa_limpia", "Total_Num", "archivo_origen"}.issubset(gasto_df.columns)


def test_persist_datasets_creates_expected_files():
    raw_path, parquet_path = persist_datasets()

    assert Path(raw_path).exists()
    assert Path(parquet_path).exists()
    assert PARTITIONED_ROOT.exists()
    assert FEDERADOS_PARQUET.exists()
    assert GASTO_PARQUET.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 17

    partitioned = pd.read_parquet(PARTITIONED_ROOT)
    assert len(partitioned) == 17 * 3

    federados = pd.read_parquet(FEDERADOS_PARQUET)
    gasto = pd.read_parquet(GASTO_PARQUET)
    assert "ccaa_limpia" in federados.columns
    assert "ccaa_limpia" in gasto.columns
