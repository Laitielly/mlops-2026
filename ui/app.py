"""
Streamlit UI — новостной классификатор.
Работает только через HTTP к serving endpoint.
"""

import os
import sys
import time
import requests
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config import cfg

st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered",
)

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


DEFAULT_URL = os.getenv("SERVING_URL", cfg.ui.default_serving_url)
TIMEOUT     = cfg.ui.request_timeout

CLASS_EMOJIS = {
    "World":    "🌍",
    "Sports":   "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬",
}

EXAMPLES = [
    "The stock market crashed after the Federal Reserve raised interest rates.",
    "NASA's Artemis mission successfully lands astronauts on the Moon.",
    "Real Madrid defeats Barcelona in a thrilling El Clasico match.",
    "World leaders gather in Geneva for emergency climate summit.",
    "Apple unveils new AI-powered chip for the iPhone 17.",
    "Olympics committee bans athlete for doping violations.",
]


with st.sidebar:
    st.header("⚙️ Настройки")
    endpoint_url = st.text_input(
        "Serving Endpoint URL",
        value=DEFAULT_URL,
        help="URL ClearML Serving endpoint",
    )

    st.divider()
    st.markdown("**Статус endpoint**")

    if st.button("🔄 Проверить"):
        try:
            health_url = endpoint_url.rsplit("/serve", 1)[0] + "/health"
            r = requests.get(health_url, timeout=3)
            if r.status_code == 200:
                info = r.json()
                st.success("✅ Online")
                st.caption(f"Model: `{info.get('model_id', 'unknown')[:8]}...`")
            else:
                st.error(f"⚠️ HTTP {r.status_code}")
        except Exception:
            st.error("❌ Недоступен")

    st.divider()
    st.markdown(
        "<small>MLOps проект · ClearML + Streamlit  \n"
        "Модель: TF-IDF + LogReg  \n"
        "Датасет: AG News</small>",
        unsafe_allow_html=True,
    )


st.title("📰 News Topic Classifier")
st.markdown(
    "Классификация тематики: "
    "**🌍 World** · **⚽ Sports** · **💼 Business** · **🔬 Sci/Tech**  \n"
    "*Инференс через HTTP endpoint*"
)
st.divider()


st.subheader("💡 Быстрые примеры")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(
        ex[:45] + "..." if len(ex) > 45 else ex,
        key=f"ex_{i}",
        use_container_width=True,
    ):
        st.session_state.input_text = ex
        st.rerun()


st.subheader("✍️ Введите текст новости")
user_text = st.text_area(
    label="text",
    value=st.session_state.input_text,
    height=130,
    placeholder="Введите текст новости на английском...",
    label_visibility="collapsed",
)
st.session_state.input_text = user_text

predict_btn = st.button(
    "🔍 Predict",
    type="primary",
    use_container_width=True,
)


def call_endpoint(text: str, url: str) -> tuple[dict, float]:
    t0       = time.perf_counter()
    response = requests.post(url, json={"text": text}, timeout=TIMEOUT)
    latency  = (time.perf_counter() - t0) * 1000
    response.raise_for_status()
    return response.json(), latency


if predict_btn:
    text = st.session_state.input_text.strip()

    if not text:
        st.warning("⚠️ Введите текст для классификации.")
    else:
        with st.spinner("Отправляем запрос..."):
            try:
                result, latency = call_endpoint(text, endpoint_url)

                label      = result.get("label", "Unknown")
                confidence = result.get("confidence", 0.0)
                probs      = result.get("probabilities", {})
                emoji      = CLASS_EMOJIS.get(label, "❓")

                st.divider()
                st.subheader("📊 Результат")

                c1, c2, c3 = st.columns(3)
                c1.metric("Тематика",    f"{emoji} {label}")
                c2.metric("Уверенность", f"{confidence * 100:.1f}%")
                c3.metric("⏱️ Latency",  f"{latency:.1f} ms")

                if probs:
                    st.subheader("📈 Вероятности")
                    for cls, prob in sorted(
                        probs.items(), key=lambda x: x[1], reverse=True
                    ):
                        st.markdown(
                            f"**{CLASS_EMOJIS.get(cls,'')} {cls}** "
                            f"— {prob * 100:.1f}%"
                        )
                        st.progress(float(prob))

                with st.expander("🔧 Raw JSON"):
                    st.json(result)
                    st.caption(f"{latency:.1f} ms · {endpoint_url}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ **Endpoint недоступен**: `{endpoint_url}`  \n"
                    "Запустите: `python serving/serve.py`"
                )
            except requests.exceptions.Timeout:
                st.error(f"⏰ Таймаут ({TIMEOUT}s) — сервер не ответил.")
            except requests.exceptions.HTTPError as e:
                st.error(f"🚫 HTTP {e.response.status_code}: {e.response.text[:200]}")
            except Exception as e:
                st.error(f"💥 Ошибка: {e}")
