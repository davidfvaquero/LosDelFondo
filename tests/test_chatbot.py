import pandas as pd

from dashboard.chatbot import generate_chat_response, normalize, prepare_assistant_data


LABELS = {
    "chat_error_data": "Sin datos.",
    "chat_max_spend": "Max spend: {region} {value}",
    "chat_min_spend": "Min spend: {region} {value}",
    "chat_max_lic": "Top licenses: {region} {value}",
    "chat_single_region": "{region}: gasto {gasto}, licencias {lic}",
    "chat_analyze": "Analizado.",
}


def sample_df() -> pd.DataFrame:
    raw_df = pd.DataFrame(
        {
            "CCAA": [
                "Madrid, Comunidad de",
                "Comunitat Valenciana",
                "Andalucía",
            ],
            "Gasto_Promedio_Hogar_Eur": [350.0, 280.0, 330.0],
            "Licencias_Federadas": [80000, 65000, 90000],
        }
    )
    return prepare_assistant_data(raw_df)


def test_normalize_removes_accents():
    assert normalize("Andalucía") == "andalucia"


def test_generate_chat_response_for_max_spend():
    response = generate_chat_response("quien gasta mas", sample_df(), LABELS)
    assert "Madrid, Comunidad de" in response


def test_generate_chat_response_for_region_alias():
    response = generate_chat_response("federados en valencia", sample_df(), LABELS)
    assert "Comunitat Valenciana" in response


def test_generate_chat_response_fallback_is_deterministic():
    response = generate_chat_response("cuentame algo", sample_df(), LABELS)
    assert response == "Analizado. Andalucía: gasto 330.0, licencias 90000"
