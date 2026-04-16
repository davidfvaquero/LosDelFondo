import os
import glob
import pandas as pd

base_dir = r"d:\ProyectoFinal\LosDelFondo\data\raw"
subdirs = [
    "Deporte_Federado",
    "Empleo_Deporte",
    "Empresa_Deporte",
    "Gasto_Hogares",
    "Gasto_Publico",
    "Habitos_Deportivos",
    "Indicadores_Afiliacion",
    "Indicadore_Armonizacion_EU"
]

results = []

for sd in subdirs:
    path = os.path.join(base_dir, sd)
    # Recursively find the first CSV
    files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if files:
        f = files[0]
        try:
            # Try to read with semicolon and latin1 (standard for these datasets)
            df = pd.read_csv(f, sep=';', encoding='latin1', nrows=1)
            cols = [c.replace('\ufeff', '').strip() for c in df.columns]
            results.append(f"{sd}: {cols}")
        except Exception as e:
            results.append(f"{sd}: Error reading {os.path.basename(f)} - {e}")
    else:
        results.append(f"{sd}: No CSV found")

for res in results:
    print(res)
