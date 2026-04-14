import pandas as pd

EXCLUDED_CCAA = {'TOTAL', 'Sin territorializar', 'Ceuta', 'Melilla'}
CCAA_KEY_MAP = {
    'asturias, principado de':     'asturias (principado de)',
    'balears, illes':              'balears (illes)',
    'madrid, comunidad de':        'madrid (comunidad de)',
    'murcia, region de':           'murcia (region de)',
    'navarra, comunidad foral de': 'navarra (comunidad foral de)',
    'rioja, la':                   'rioja (la)',
}

fed = pd.read_parquet('data/processed/federados.parquet')
gas = pd.read_parquet('data/processed/gasto.parquet')

for yr in [2006, 2015, 2023]:
    fed_year = (
        fed[
            (fed['periodo'] == yr)
            & (fed['Federacion'] == 'TOTAL' if 'Federacion' in fed.columns else fed['Federaci\u00f3n'] == 'TOTAL')
            & (~fed['Comunidad aut\u00f3noma'].isin(EXCLUDED_CCAA))
        ][['Comunidad aut\u00f3noma', 'ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Licencias_Federadas', 'Comunidad aut\u00f3noma': 'CCAA'})
        .copy()
    )
    fed_col = 'Federaci\u00f3n' if 'Federaci\u00f3n' in fed.columns else 'Federacion'
    fed_year = (
        fed[
            (fed['periodo'] == yr)
            & (fed[fed_col] == 'TOTAL')
            & (~fed['Comunidad aut\u00f3noma'].isin(EXCLUDED_CCAA))
        ][['Comunidad aut\u00f3noma', 'ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Licencias_Federadas', 'Comunidad aut\u00f3noma': 'CCAA'})
        .copy()
    )
    fed_year['ccaa_key'] = fed_year['ccaa_limpia'].map(lambda x: CCAA_KEY_MAP.get(x, x))

    gas_year = (
        gas[
            (gas['periodo'] == yr)
            & (gas['Indicador'] == 'Gasto medio por hogar (Euros)')
            & (gas['Comunidad aut\u00f3noma'] != 'TOTAL')
        ][['ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Gasto_Promedio_Hogar_Eur', 'ccaa_limpia': 'ccaa_key'})
    )

    df = pd.merge(fed_year, gas_year, on='ccaa_key', how='inner').drop(columns=['ccaa_limpia', 'ccaa_key'])
    print(f"Anio {yr}: {len(df)} CCAA  |  gasto_avg={df['Gasto_Promedio_Hogar_Eur'].mean():.1f}  |  licencias={df['Licencias_Federadas'].sum():,.0f}")
    print("  CCAA:", sorted(df['CCAA'].tolist()))
    print()
