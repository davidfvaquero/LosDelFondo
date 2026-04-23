from __future__ import annotations

from pathlib import Path
import re
import shutil
import unicodedata

import pandas as pd


SOURCE_YEAR = 2023
SOURCE_NAME = "gasto_y_federado_2023"

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT_DIR / "data" / "raw" / f"{SOURCE_NAME}.csv"
FEDERADOS_CSV = ROOT_DIR / "data" / "raw" / "federado_01bsc.csv"
GASTO_CSV = ROOT_DIR / "data" / "raw" / "gasto_03bsc.csv"
PROCESSED_ROOT = ROOT_DIR / "data" / "processed"
FEDERADOS_PARQUET = PROCESSED_ROOT / "federados.parquet"
GASTO_PARQUET = PROCESSED_ROOT / "gasto.parquet"
PROCESSED_DIR = PROCESSED_ROOT / "deporte_data" / f"anio={SOURCE_YEAR}"
PROCESSED_PARQUET = PROCESSED_DIR / "hechos_indicadores.parquet"
PARTITIONED_ROOT = PROCESSED_ROOT / "deporte_data_partitioned"

INDICATOR_COLUMNS = {
    "Gasto_Promedio_Hogar_Eur": "gasto_promedio_hogar_eur",
    "Licencias_Federadas": "licencias_federadas",
    "Poblacion_Activa_Dep": "poblacion_activa_dep",
}


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")


def normalize_ccaa(value: str) -> str:
    return str(value).strip().lower()


def reset_output_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def generate_source_dataframe() -> pd.DataFrame:
    return pd.read_csv(SOURCE_CSV)


def build_processed_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    processed_df = source_df.copy()
    processed_df["anio"] = SOURCE_YEAR
    processed_df["fuente"] = SOURCE_NAME
    return processed_df


def build_partitioned_dataframe(processed_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for _, row in processed_df.iterrows():
        ccaa = row["CCAA"]
        ccaa_slug = slugify(ccaa)
        for source_column, indicador_slug in INDICATOR_COLUMNS.items():
            records.append(
                {
                    "anio": SOURCE_YEAR,
                    "fuente": SOURCE_NAME,
                    "CCAA": ccaa,
                    "ccaa": ccaa_slug,
                    "indicador": indicador_slug,
                    "valor": row[source_column],
                }
            )
    return pd.DataFrame.from_records(records)


def build_federados_dataframe() -> pd.DataFrame:
    df = pd.read_csv(FEDERADOS_CSV, sep=";", encoding="ISO-8859-1")
    df = df[df["Comunidad autónoma"].notna()].copy()
    df["ccaa_limpia"] = df["Comunidad autónoma"].map(normalize_ccaa)
    df["archivo_origen"] = "federado_01.csv"
    cleaned = (
        df["Total"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .replace({"": pd.NA})
    )
    df["Total_Num"] = pd.to_numeric(cleaned, errors="coerce").astype("Int64")
    return df


def build_gasto_dataframe() -> pd.DataFrame:
    df = pd.read_csv(GASTO_CSV, sep=";", encoding="ISO-8859-1")
    df = df[df["Comunidad autónoma"].notna()].copy()
    df["ccaa_limpia"] = df["Comunidad autónoma"].map(normalize_ccaa)
    df["archivo_origen"] = "gasto_03.csv"
    cleaned = (
        df["Total"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": pd.NA})
    )
    df["Total_Num"] = pd.to_numeric(cleaned, errors="coerce")
    return df


def persist_raw_partitioned_parquets() -> tuple[str, str]:
    federados_df = build_federados_dataframe()
    gasto_df = build_gasto_dataframe()

    reset_output_path(FEDERADOS_PARQUET)
    reset_output_path(GASTO_PARQUET)

    federados_df.to_parquet(
        FEDERADOS_PARQUET,
        index=False,
        partition_cols=["periodo", "ccaa_limpia"],
    )
    gasto_df.to_parquet(
        GASTO_PARQUET,
        index=False,
        partition_cols=["periodo", "ccaa_limpia"],
    )

    return str(FEDERADOS_PARQUET), str(GASTO_PARQUET)


def persist_analytic_datasets() -> tuple[str, str]:
    source_df = generate_source_dataframe()
    processed_df = build_processed_dataframe(source_df)
    partitioned_df = build_partitioned_dataframe(processed_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(PROCESSED_PARQUET, index=False)

    reset_output_path(PARTITIONED_ROOT)
    partitioned_df.to_parquet(
        PARTITIONED_ROOT,
        index=False,
        partition_cols=["anio", "ccaa", "indicador"],
    )

    return str(SOURCE_CSV), str(PROCESSED_PARQUET)


def persist_datasets() -> tuple[str, str]:
    persist_raw_partitioned_parquets()
    return persist_analytic_datasets()


def main() -> None:
    source_path, parquet_path = persist_datasets()
    print(f"Fuente cargada: {source_path}")
    print(f"Parquet generado: {parquet_path}")
    print(f"Particiones generadas: {PARTITIONED_ROOT}")
    print(f"Federados parquet: {FEDERADOS_PARQUET}")
    print(f"Gasto parquet: {GASTO_PARQUET}")


if __name__ == "__main__":
    main()
