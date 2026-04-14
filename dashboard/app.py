import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px


try:
    from dashboard.chatbot import (
        prepare_assistant_data,
        check_toxicity, load_models, generate_llm_response,
    )
except ImportError:
    from chatbot import (
        prepare_assistant_data,
        check_toxicity, load_models, generate_llm_response,
    )

import unicodedata

def normalize(text: str) -> str:
    """Lowercase without accents."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )

MANUAL_TOXIC_TERMS = [
    "puta", "puto", "putos", "putas",
    "hijo de puta", "hija de puta",
    "inutil", "imbecil", "idiota", "gilipollas",
    "capullo", "mamona", "mamon", "culo",
    "mierda", "hostia", "joder", "cono",
    "pendejo", "cabron", "cabrona", "zorra",
    "fuck", "shit", "asshole", "bitch", "damn",
]

# Inicialización de estado de sesión
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'Oscuro'
if 'lang' not in st.session_state:
    st.session_state.lang = 'ES'
if 'sel_year' not in st.session_state:
    st.session_state.sel_year = '2023'
if 'sel_territory' not in st.session_state:
    st.session_state.sel_territory = 'Todas las CCAA'
if 'sel_territory_input' not in st.session_state:
    st.session_state.sel_territory_input = 'Todas las CCAA'

# Diccionario de Traducciones
LANGUAGES = {
    'ES': {
        'page_title': "DEPORTEData | Reto A",
        'sidebar_prefs': "Preferencias",
        'sidebar_theme': "Modo Visualización",
        'sidebar_lang': "Idioma",
        'sidebar_filters': "Seleccionar Filtros Genéricos",
        'filter_year': "Año de Datos",
        'filter_territory': "Territorio",
        'all_ccaa': "Todas las CCAA",
        'login_admin': "🔓 Login Admin",
        'admin_label': "👤 Administrador",
        'main_title': "DEPORTEData: Analítica e Inteligencia Deportiva",
        'main_subtitle': "Evolución del gasto en deporte por hogar y su relación con la práctica deportiva federada por CCAA",
        'login_header': "🔑 Inicio de Sesión - Administrador",
        'login_user': "Usuario",
        'login_pass': "Contraseña",
        'login_btn': "Entrar",
        'login_err': "Credenciales incorrectas",
        'tab_home': "📊 Inicio",
        'tab_ai': "🤖 Asistente IA (RAG)",
        'tab_admin': "⚙️ Admin",
        'dash_header': "Dashboard Analítico",
        'metric_spending': "Gasto Promedio/Hogar",
        'metric_licenses': "Licencias Federadas",
        'metric_areas': "Áreas Analizadas",
        'chart_evolution': "Evolución de Gasto vs Licencias",
        'chart_spending_region': "Gasto Promedio por Hogar por CCAA",
        'table_indicators': "Tabla de Indicadores",
        'err_no_data': "No se encontraron datos para los filtros seleccionados o el archivo no existe.",
        'chat_header': "Consulta Inteligente sobre DEPORTEData",
        'chat_input': "Escribe tu pregunta aquí...",
        'chat_hi': "¡Hola! Soy el asistente IA de DEPORTEData. ¿En qué puedo ayudarte?",
        'chat_max_spend': "🔍 La CCAA que más gasta es {region} con {value} €.",
        'chat_min_spend': "🔍 La CCAA que menos gasta es {region} con {value} €.",
        'chat_max_lic': "🏆 {region} lidera en licencias con {value:,}.",
        'chat_single_region': "📍 En {region}: Gasto de {gasto} € y {lic} licencias.",
        'chat_analyze': "🧠 He analizado los datos actuales. ¿Deseas algún detalle específico?",
        'chat_error_data': "⚠️ No hay datos disponibles para el análisis.",
        'admin_header': "Panel de Administración",
        'admin_usage': "📊 Panel de Uso",
        'admin_active': "Usuarios Activos (24h)",
        'admin_queries': "Consultas Totales",
        'admin_security': "🛡️ Seguridad y Accesos",
        'admin_failed': "Intentos Fallidos (24h)",
        'admin_last_logs': "Últimos Intentos de Acceso",
        'admin_telemetry': "📡 Telemetría de Sistema",
        'admin_cpu': "Carga CPU",
        'admin_ram': "Uso RAM",
        'admin_system_load': "Carga de Sistema (20 min)",
        'col_gasto': "Gasto Promedio Hogar Eur",
        'col_lic': "Licencias Federadas",
        'col_ccaa': "CCAA",
        'chart_q': "Consultas IA",
        'chart_v': "Visitas Dashboard",
        'chart_l': "Carga"
    },
    'EN': {
        'page_title': "DEPORTEData | Challenge A",
        'sidebar_prefs': "Preferences",
        'sidebar_theme': "Display Mode",
        'sidebar_lang': "Language",
        'sidebar_filters': "Select General Filters",
        'filter_year': "Data Year",
        'filter_territory': "Territory",
        'all_ccaa': "All Regions",
        'login_admin': "🔓 Admin Login",
        'admin_label': "👤 Administrator",
        'main_title': "DEPORTEData: Sports Analytics & Intelligence",
        'main_subtitle': "Evolution of household sports spending and its relationship with federated sports practice by region",
        'login_header': "🔑 Admin Login",
        'login_user': "Username",
        'login_pass': "Password",
        'login_btn': "Login",
        'login_err': "Incorrect credentials",
        'tab_home': "📊 Home",
        'tab_ai': "🤖 AI Assistant (RAG)",
        'tab_admin': "⚙️ Admin",
        'dash_header': "Analytical Dashboard",
        'metric_spending': "Avg Spending/Home",
        'metric_licenses': "Federated Licenses",
        'metric_areas': "Analyzed Areas",
        'chart_evolution': "Spending vs Licenses Evolution",
        'chart_spending_region': "Avg Household Spending by Region",
        'table_indicators': "Indicators Table",
        'err_no_data': "No data found for the selected filters or file does not exist.",
        'chat_header': "Intelligent Query about DEPORTEData",
        'chat_input': "Type your question here...",
        'chat_hi': "Hi! I am the DEPORTEData AI assistant. How can I help you?",
        'chat_max_spend': "🔍 The region with the highest spending is {region} with {value} €.",
        'chat_min_spend': "🔍 The region with the lowest spending is {region} with {value} €.",
        'chat_max_lic': "🏆 {region} leads in licenses with {value:,}.",
        'chat_single_region': "📍 In {region}: Spending of {gasto} € and {lic} licenses.",
        'chat_analyze': "🧠 I have analyzed the current data. Do you need any specific details?",
        'chat_error_data': "⚠️ No data available for analysis.",
        'admin_header': "Administration Panel",
        'admin_usage': "📊 Usage Panel",
        'admin_active': "Active Users (24h)",
        'admin_queries': "Total Queries",
        'admin_security': "🛡️ Security & Access",
        'admin_failed': "Failed Attempts (24h)",
        'admin_last_logs': "Last Access Attempts",
        'admin_telemetry': "📡 System Telemetry",
        'admin_cpu': "CPU Load",
        'admin_ram': "RAM Usage",
        'admin_system_load': "System Load (20 min)",
        'col_gasto': "Avg Household Spending Eur",
        'col_lic': "Federated Licenses",
        'col_ccaa': "Region",
        'chart_q': "AI Queries",
        'chart_v': "Dashboard Visits",
        'chart_l': "Load"
    }
}
L = LANGUAGES[st.session_state.lang]

# Configuración de la página
icon_path = "dashboard/icon_white.png" if st.session_state.theme == 'Oscuro' else "dashboard/icon.png"

st.set_page_config(
    page_title=L['page_title'],
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados dinámicos para Accesibilidad Total
if st.session_state.theme == 'Oscuro':
    bg_color = "#0E1117"
    text_color = "#FFFFFF"
    accent_color = "#00E676"  # Verde neón
    sidebar_bg = "#262730"
    plotly_template = "plotly_dark"
    table_border = "#444444"
    chart_axis_color = "#CCCCCC"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    accent_color = "#0066CC"  # Azul cobalto
    sidebar_bg = "#F0F2F6"
    plotly_template = "plotly_white"
    table_border = "#DDDDDD"
    chart_axis_color = "#333333"

# Inyectar CSS Dinámico
st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    header, [data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
    footer {{
        visibility: hidden;
    }}
    .main .block-container {{
        padding-top: 1rem;
    }}
    h1, h2, h3, h4, h5, h6, .stMetric label {{
        color: {accent_color} !important;
    }}
    .stMarkdown, p, span, div, label, .stMetricValue {{
        color: {text_color} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {text_color} !important;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        color: {text_color} !important;
        background-color: {bg_color} !important;
    }}
    th {{
        background-color: {sidebar_bg} !important;
        color: {accent_color} !important;
        border: 1px solid {table_border} !important;
        padding: 8px;
        text-align: left;
    }}
    td {{
        border: 1px solid {table_border} !important;
        padding: 8px;
        color: {text_color} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {bg_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {text_color};
    }}
    .stChatMessage {{
        background-color: {sidebar_bg} !important;
        border-radius: 10px;
        margin-bottom: 10px;
    }}
    [data-testid="stTextInputRootElement"] {{
        background-color: {"#FFFFFF" if st.session_state.theme == "Claro" else "#1A1C23"} !important;
        border: 1px solid {table_border} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stTextInputRootElement"] > div {{
        background-color: {"#FFFFFF" if st.session_state.theme == "Claro" else "#1A1C23"} !important;
    }}
    [data-testid="stTextInputRootElement"] input {{
        background-color: transparent !important;
        color: {text_color} !important;
    }}
    [data-testid="stTextInputRootElement"] input::placeholder {{
        color: {text_color} !important;
        opacity: 0.6 !important;
    }}
    [data-testid="stTextInputRootElement"] svg {{
        fill: {text_color} !important;
    }}

    /* Refinamiento Chat Input (Chatbot) */
    [data-testid="stChatInput"] {{
        position: sticky !important;
        bottom: 0 !important;
        padding-bottom: 2rem !important;
        padding-top: 1rem !important;
        background-color: {bg_color} !important;
        z-index: 99 !important;
        border: none !important;
    }}
    [data-testid="stChatInput"] div {{
        background-color: transparent !important;
        border: none !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: {"#FFFFFF" if st.session_state.theme == "Claro" else "#1A1C23"} !important;
        color: {text_color} !important;
        border: 1px solid {table_border} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {text_color} !important;
        opacity: 0.6;
    }}
    [data-testid="stChatInput"] button {{
        background-color: {accent_color} !important;
        color: {"#000000" if st.session_state.theme == "Oscuro" else "#FFFFFF"} !important;
        border-radius: 5px !important;
    }}

    /* Estilo para los selectores (filtros) - Hacerlos 'claritos' en modo claro */
    div[data-baseweb="select"] > div {{
        background-color: {"#FFFFFF" if st.session_state.theme == "Claro" else "#1A1C23"} !important;
        color: {text_color} !important;
        border: 1px solid {table_border} !important;
    }}
    
    /* Corregir a fondo Menu de Configuración (3 puntos) en modo claro */
    div[data-baseweb="popover"] ul, 
    div[data-baseweb="popover"] li, 
    div[data-baseweb="popover"] div, 
    div[data-testid="stPopoverBody"], 
    ul[data-testid="main-menu-list"], 
    ul[data-testid="main-menu-list"] li,
    ul[data-testid="main-menu-list"] div {{
        background-color: {"#FFFFFF" if st.session_state.theme == "Claro" else "#1A1C23"} !important;
        color: {text_color} !important;
    }}
    
    div[data-baseweb="popover"] span, 
    ul[data-testid="main-menu-list"] span,
    div[data-baseweb="popover"] p {{
        color: {text_color} !important;
    }}
    
    /* Efecto hover en el menu */
    div[data-baseweb="popover"] li:hover, 
    ul[data-testid="main-menu-list"] li:hover {{
        background-color: {"#F0F2F6" if st.session_state.theme == "Claro" else "#262730"} !important;
    }}
    div[data-baseweb="select"] ul {{
        background-color: {"#FFFFFF" if st.session_state.theme == "Claro" else "#1A1C23"} !important;
    }}
    div[data-baseweb="select"] li {{
        color: {text_color} !important;
    }}
    div[data-baseweb="radio"] label,
    div[data-baseweb="radio"] div {{
        color: {text_color} !important;
    }}
    div.stButton > button, [data-testid="stFormSubmitButton"] > button, div.stButton > button p, [data-testid="stFormSubmitButton"] > button p {{
        color: {"#000000" if st.session_state.theme == "Oscuro" else "#FFFFFF"} !important;
    }}
    div.stButton > button, [data-testid="stFormSubmitButton"] > button {{
        background-color: {accent_color} !important;
        border: none !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }}
</style>
""", unsafe_allow_html=True)

# Helper para configurar gráficos de Plotly
def apply_plotly_style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=text_color,
        template=plotly_template,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)", tickfont=dict(color=chart_axis_color), title=dict(font=dict(color=text_color))),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)", tickfont=dict(color=chart_axis_color), title=dict(font=dict(color=text_color))),
        legend=dict(font=dict(color=text_color))
    )
    return fig


@st.cache_data
def load_home_data(year: str) -> pd.DataFrame:
    """Carga y une federados.parquet + gasto.parquet para el año indicado.
    federados usa comas ('asturias, principado de') y gasto usa paréntesis
    ('asturias (principado de)'), por lo que se normaliza ccaa_limpia antes del join."""
    EXCLUDED_CCAA = {'TOTAL', 'Sin territorializar', 'Ceuta', 'Melilla'}

    # Mapeo ccaa_limpia de federados → clave equivalente en gasto
    CCAA_KEY_MAP = {
        'asturias, principado de':     'asturias (principado de)',
        'balears, illes':              'balears (illes)',
        'madrid, comunidad de':        'madrid (comunidad de)',
        'murcia, regi\u00f3n de':           'murcia (regi\u00f3n de)',
        'navarra, comunidad foral de': 'navarra (comunidad foral de)',
        'rioja, la':                   'rioja (la)',
    }

    fed = pd.read_parquet("data/processed/federados.parquet")
    gas = pd.read_parquet("data/processed/gasto.parquet")
    yr = int(year)

    # Licencias totales por CCAA (fila 'TOTAL' de federación)
    fed_year = (
        fed[
            (fed['periodo'] == yr)
            & (fed['Federación'] == 'TOTAL')
            & (~fed['Comunidad autónoma'].isin(EXCLUDED_CCAA))
        ][['Comunidad autónoma', 'ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Licencias_Federadas', 'Comunidad autónoma': 'CCAA'})
        .copy()
    )
    # Normalizar ccaa_limpia de federados para que coincida con gasto
    fed_year['ccaa_key'] = fed_year['ccaa_limpia'].map(
        lambda x: CCAA_KEY_MAP.get(x, x)
    )

    # Gasto medio por hogar
    gas_year = (
        gas[
            (gas['periodo'] == yr)
            & (gas['Indicador'] == 'Gasto medio por hogar (Euros)')
            & (gas['Comunidad autónoma'] != 'TOTAL')
        ][['ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Gasto_Promedio_Hogar_Eur', 'ccaa_limpia': 'ccaa_key'})
    )

    df = pd.merge(fed_year, gas_year, on='ccaa_key', how='inner')
    df = df.drop(columns=['ccaa_limpia', 'ccaa_key'])
    return df

# Título y encabezado
st.title(L['main_title'])
st.markdown(f"### {L['main_subtitle']}")

# Pantalla de login
if st.session_state.show_login:
    st.header(L['login_header'])
    with st.form("login_form"):
        username = st.text_input(L['login_user'])
        password = st.text_input(L['login_pass'], type="password")
        submit = st.form_submit_button(L['login_btn'])
        if submit:
            if username == "admin" and password == "1234":
                st.session_state.is_admin = True
                st.session_state.show_login = False
                st.rerun()
            else:
                st.error(L['login_err'])
    st.stop()

# Definición de pestañas
tabs_list = [L['tab_home'], L['tab_ai']]
if st.session_state.is_admin:
    tabs_list.append(L['tab_admin'])
tabs = st.tabs(tabs_list)
tab1, tab2 = tabs[0], tabs[1]
if st.session_state.is_admin:
    tab_admin = tabs[2]

with tab1:
    st.header(f"{L['dash_header']} - {st.session_state.sel_year}")
    try:
        # Cargar datos desde federados.parquet y gasto.parquet
        df_real = load_home_data(st.session_state.sel_year)
        df_display = df_real.rename(columns={
            'Gasto_Promedio_Hogar_Eur': L['col_gasto'],
            'Licencias_Federadas': L['col_lic'],
            'CCAA': L['col_ccaa'],
        })

        # Aplicar filtro de Territorio
        if st.session_state.sel_territory != "Todas las CCAA":
            df_filtered = df_display[df_real['CCAA'] == st.session_state.sel_territory]
        else:
            df_filtered = df_display

        # Métricas Dinámicas
        col1, col2, col3 = st.columns(3)
        if not df_filtered.empty:
            avg_gasto = df_filtered[L['col_gasto']].mean()
            total_licencias = df_filtered[L['col_lic']].sum()
            num_ccaa = len(df_filtered)

            col1.metric(L['metric_spending'], f"€ {avg_gasto:.0f}", None)
            col2.metric(
                L['metric_licenses'],
                f"{total_licencias/1e6:.1f}M" if total_licencias > 1e5 else f"{total_licencias:,.0f}",
                None,
            )
            col3.metric(L['metric_areas'], str(num_ccaa), None)

        st.subheader(L['chart_evolution'])
        fig_scatter = px.scatter(
            df_filtered, x=L['col_gasto'], y=L['col_lic'],
            hover_name=L['col_ccaa'], color_discrete_sequence=[accent_color],
        )
        st.plotly_chart(apply_plotly_style(fig_scatter), use_container_width=True)

        st.subheader(L['chart_spending_region'])
        fig_bar = px.bar(
            df_filtered.sort_values(L['col_gasto'], ascending=False),
            x=L['col_ccaa'], y=L['col_gasto'],
            color_discrete_sequence=[accent_color],
        )
        st.plotly_chart(apply_plotly_style(fig_bar), use_container_width=True)

        st.subheader(L['table_indicators'])
        df_table = df_filtered[[L['col_ccaa'], L['col_gasto'], L['col_lic']]].copy()
        df_table.index = range(1, len(df_table) + 1)
        st.table(df_table)

    except Exception as e:
        st.error(f"{L['err_no_data']} ({e})")

with tab2:
    st.header(L['chat_header'])
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": L['chat_hi']}]
        
    msg_container = st.container()
    
    with msg_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    if prompt := st.chat_input(L['chat_input']):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with msg_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
            full_response = ""

            # ── Layer 1: manual toxic check (no model needed) ──────────────
            blocked_msg_es = "⚠️ El mensaje ha sido bloqueado por nuestra política de seguridad debido a lenguaje tóxico o inapropiado."
            blocked_msg_en = "⚠️ The message has been blocked by our security policy due to toxic or inappropriate language."
            blocked_msg = blocked_msg_es if st.session_state.lang == "ES" else blocked_msg_en

            prompt_norm = normalize(prompt)
            manually_toxic = any(normalize(t) in prompt_norm for t in MANUAL_TOXIC_TERMS)

            if manually_toxic:
                assistant_response = blocked_msg
            else:
                # ── Layer 2: load data + try AI pipeline ──────────────────
                try:
                    @st.cache_data
                    def load_assistant_data():
                        df = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
                        return prepare_assistant_data(df)

                    df_rag = load_assistant_data()

                    # Try AI model
                    try:
                        toxic_clf, llm_pipeline = load_models()
                        is_toxic, _ = check_toxicity(prompt, toxic_clf)
                        if is_toxic:
                            assistant_response = blocked_msg
                        else:
                            assistant_response = generate_llm_response(
                                prompt, df_rag, llm_pipeline, st.session_state.lang
                            )
                    except Exception as ai_err:
                        # Show the real error so we can diagnose it
                        assistant_response = f"❌ Error al cargar el modelo IA: {type(ai_err).__name__}: {ai_err}"

                except Exception as e:
                    assistant_response = f"{L['chat_error_data']} ({str(e)})"

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.session_state.is_admin:
    with tab_admin:
        st.header(L['admin_header'])
        st.subheader(L['admin_usage'])
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.metric(L['admin_active'], "142", None)
            fig_usage = px.line(pd.DataFrame(np.random.randn(20, 2), columns=[L['chart_q'], L['chart_v']]), color_discrete_sequence=[accent_color, "#FF4B4B"])
            st.plotly_chart(apply_plotly_style(fig_usage), use_container_width=True)
        with col_u2:
            st.metric(L['admin_queries'], "2,840", None)
            fig_total = px.bar(np.random.randint(10, 100, size=(7, 1)), color_discrete_sequence=[accent_color])
            st.plotly_chart(apply_plotly_style(fig_total), use_container_width=True)
        
        st.divider()
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader(L['admin_security'])
            st.metric(L['admin_failed'], "3", "-12%", delta_color="normal")
            st.markdown(f"**{L['admin_last_logs']}**")
            admin_logs = pd.DataFrame({
                "User": ["admin", "root", "guest", "admin"],
                "IP": ["192.168.1.45", "85.23.11.102", "172.16.0.5", "192.168.1.45"],
                "Status": ["Success", "Blocked", "Failed", "Success"]
            })
            st.table(admin_logs)
            
        with col_s2:
            st.subheader(L['admin_telemetry'])
            col_t1, col_t2 = st.columns(2)
            col_t1.metric(L['admin_cpu'], "24%", "2%")
            col_t2.metric(L['admin_ram'], "1.2 GB", "0.1 GB")
            st.markdown(f"**{L['admin_system_load']}**")
            telemetry_data = pd.DataFrame(np.random.randn(20, 1), columns=[L['chart_l']])
            fig_telemetry = px.area(telemetry_data, color_discrete_sequence=[accent_color])
            st.plotly_chart(apply_plotly_style(fig_telemetry), use_container_width=True)

# Sidebar y personalización
with st.sidebar:
    st.image(icon_path, width=100)
    st.markdown("## DEPORTEData")
    st.divider()
    
    st.markdown(f"### {L['sidebar_prefs']}")
    
    # Selector de Idioma
    lang = st.selectbox(
        L['sidebar_lang'], 
        ["ES", "EN"], 
        index=0 if st.session_state.lang == "ES" else 1,
        format_func=lambda x: "🇪🇸 Español" if x == "ES" else "🇺🇸 English"
    )
    if lang != st.session_state.lang:
        st.session_state.lang = lang
        # Refrescar saludo del chatbot al cambiar idioma
        st.session_state.messages = [{"role": "assistant", "content": LANGUAGES[lang]['chat_hi']}]
        st.rerun()

    theme = st.radio(L['sidebar_theme'], ["Oscuro", "Claro"] if st.session_state.lang == "ES" else ["Dark", "Light"], index=0 if st.session_state.theme in ["Oscuro", "Dark"] else 1, horizontal=True)
    theme_val = "Oscuro" if theme in ["Oscuro", "Dark"] else "Claro"
    if theme_val != st.session_state.theme:
        st.session_state.theme = theme_val
        st.rerun()

    st.divider()
    
    st.markdown(f"### {L['sidebar_filters']}")
    # Filtro de Año persistente
    # Años disponibles: intersección de federados (2005-2024) y gasto (2006-2023)
    year_options = [str(y) for y in range(2023, 2005, -1)]
    sel_year_idx = year_options.index(st.session_state.sel_year) if st.session_state.sel_year in year_options else 0
    st.selectbox(
        L['filter_year'],
        year_options,
        index=sel_year_idx,
        key='sel_year'
    )

    # Filtro de Territorio persistente (nombres exactos de federados.parquet)
    territory_options = [
        L['all_ccaa'],
        "Andalucía", "Aragón", "Asturias, Principado de", "Balears, Illes",
        "Canarias", "Cantabria", "Castilla y León", "Castilla-La Mancha",
        "Cataluña", "Comunitat Valenciana", "Extremadura", "Galicia",
        "Madrid, Comunidad de", "Murcia, Región de", "Navarra, Comunidad Foral de",
        "País Vasco", "Rioja, La",
    ]
    st.selectbox(
        L['filter_territory'], 
        territory_options, 
        index=territory_options.index(st.session_state.sel_territory) if st.session_state.sel_territory in territory_options else 0,
        key='sel_territory_input'
    )
    # Sincronizar sel_territory con el nombre interno (Todas las CCAA)
    if st.session_state.sel_territory_input == L['all_ccaa']:
        st.session_state.sel_territory = "Todas las CCAA"
    else:
        st.session_state.sel_territory = st.session_state.sel_territory_input

    st.divider()
    
    if not st.session_state.is_admin:
        if st.button(L['login_admin']):
            st.session_state.show_login = True
            st.rerun()
    else:
        st.success(L['admin_label'])

