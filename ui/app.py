"""
Streamlit UI для классификации тематики новостей.
Работает через HTTP к ClearML Serving endpoint.
Модель НЕ загружается напрямую.
"""

import time
import requests
import streamlit as st


ENDPOINT_URL = st.sidebar.text_input(
    "Serving Endpoint URL",
    value="http://localhost:8080/serve/news-topic",
    help="URL ClearML Serving endpoint"
)

CLASS_NAMES  = ["World", "Sports", "Business", "Sci/Tech"]
CLASS_EMOJIS = {
    "World":    "🌍",
    "Sports":   "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬",
}
CLASS_COLORS = {
    "World":    "#4A90D9",
    "Sports":   "#27AE60",
    "Business": "#E67E22",
    "Sci/Tech": "#8E44AD",
}

EXAMPLES = [
    "The stock market crashed after the Federal Reserve raised interest rates.",
    "NASA's Artemis mission successfully lands astronauts on the Moon.",
    "Real Madrid defeats Barcelona in a thrilling El Clasico match.",
    "World leaders gather in Geneva for emergency climate summit.",
    "Apple unveils new AI-powered chip for the iPhone 17.",
    "Olympics committee bans athlete for doping violations.",
]


st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered",
)

st.title("📰 News Topic Classifier")
st.markdown(
    "Классификация тематики новостного текста: "
    "**World** · **Sports** · **Business** · **Sci/Tech**  \n"
    "*Инференс через ClearML Serving HTTP endpoint*"
)

st.divider()


st.subheader("💡 Быстрые примеры")
cols = st.columns(3)
selected_example = None
for i, ex in enumerate(EXAMPLES):
    col = cols[i % 3]
    short = ex[:45] + "..." if len(ex) > 45 else ex
    if col.button(short, key=f"ex_{i}"):
        selected_example = ex


st.subheader("✍️ Введите текст новости")

default_text = selected_example if selected_example else ""
user_text = st.text_area(
    label="Текст новости",
    value=default_text,
    height=130,
    placeholder="Введите или вставьте текст новости на английском языке...",
    label_visibility="collapsed",
)

predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

def call_endpoint(text: str, url: str) -> tuple[dict, float]:
    """
    Отправляет POST-запрос к ClearML Serving.
    Возвращает (response_dict, latency_ms).
    Бросает исключение при ошибке.
    """
    payload = {"text": text}
    t0 = time.perf_counter()
    response = requests.post(
        url,
        json=payload,
        timeout=10,
    )
    latency = (time.perf_counter() - t0) * 1000  # ms
    response.raise_for_status()
    return response.json(), latency


if predict_btn:
    if not user_text.strip():
        st.warning("⚠️ Пожалуйста, введите текст для классификации.")
    else:
        with st.spinner("Отправляем запрос к модели..."):
            try:
                result, latency = call_endpoint(user_text.strip(), ENDPOINT_URL)

                label      = result.get("label", "Unknown")
                confidence = result.get("confidence", 0.0)
                probs      = result.get("probabilities", {})
                emoji      = CLASS_EMOJIS.get(label, "❓")
                color      = CLASS_COLORS.get(label, "#333")

                st.divider()
                st.subheader("📊 Результат")

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    label="Тематика",
                    value=f"{emoji} {label}",
                )
                col2.metric(
                    label="Уверенность",
                    value=f"{confidence * 100:.1f}%",
                )
                col3.metric(
                    label="⏱️ Latency",
                    value=f"{latency:.1f} ms",
                )

                if probs:
                    st.subheader("📈 Вероятности по классам")
                    sorted_probs = sorted(
                        probs.items(), key=lambda x: x[1], reverse=True
                    )
                    for cls, prob in sorted_probs:
                        bar_color = CLASS_COLORS.get(cls, "#999")
                        st.markdown(
                            f"**{CLASS_EMOJIS.get(cls, '')} {cls}** — {prob * 100:.1f}%"
                        )
                        st.progress(float(prob))

                with st.expander("🔧 Raw JSON ответ"):
                    st.json(result)
                    st.caption(f"Latency: {latency:.2f} ms | Endpoint: {ENDPOINT_URL}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Endpoint недоступен**  \n"
                    f"Не удалось подключиться к `{ENDPOINT_URL}`.  \n"
                    "Проверьте, что ClearML Serving запущен."
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⏰ **Таймаут**  \n"
                    "Сервер не ответил за 10 секунд. Попробуйте позже."
                )
            except requests.exceptions.HTTPError as e:
                st.error(
                    f"🚫 **HTTP ошибка**: {e.response.status_code}  \n"
                    f"```{e.response.text[:300]}```"
                )
            except Exception as e:
                st.error(f"💥 **Неожиданная ошибка**: {e}")

st.divider()
st.markdown(
    "<small>MLOps курсовой проект · ClearML + Streamlit · "
    "Модель: TF-IDF + LogReg · Датасет: AG News</small>",
    unsafe_allow_html=True,
)
