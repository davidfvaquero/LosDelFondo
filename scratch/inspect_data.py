import pandas as pd
df = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
print(df[df['CCAA'].str.contains('Murcia')])
print("\nMax spend:")
print(df.loc[df['Gasto_Promedio_Hogar_Eur'].idxmax()])
print("\nMin spend:")
print(df.loc[df['Gasto_Promedio_Hogar_Eur'].idxmin()])
