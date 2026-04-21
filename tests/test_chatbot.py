import pandas as pd

from dashboard.chatbot import _fallback_generate_chat_response, normalize, prepare_assistant_data

#a
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
    response = _fallback_generate_chat_response("quien gasta mas", sample_df(), "ES")
    assert "Madrid, Comunidad de" in response


def test_generate_chat_response_for_region_alias():
    response = _fallback_generate_chat_response("federados en valencia", sample_df(), "ES")
    assert "Comunitat Valenciana" in response


def test_generate_chat_response_fallback_is_deterministic():
    response = _fallback_generate_chat_response("cuentame algo", sample_df(), "ES")
    assert "Andalucía" in response
    assert "330.0" in response
    assert "90000" in response
