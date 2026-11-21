import os, uuid
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

from src.modelo import cargar_modelo, predecir_sentimiento
from src.agente import crear_agente

st.set_page_config(page_title="Agente de triaje neuro-agéntico", page_icon="🧠", layout="wide")

@st.cache_resource
def get_modelo():
    return cargar_modelo()

@st.cache_resource
def get_agente():
    return crear_agente()

get_modelo()
agente = get_agente()

def texto(msg):
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.historial = []

st.title("🧠 Agente de triaje de reseñas")
st.caption("U4: red neuronal Embedding+Dense (IMDB) · U5: agente LangGraph con 2 herramientas y memoria")

with st.sidebar:
    st.subheader("Sesión")
    st.code(f"thread_id: {st.session_state.thread_id}")
    if st.button("🔄 Reiniciar sesión (nuevo thread_id)"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.historial = []
        st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Red neuronal (salida directa U4)")
    resena = st.text_area("Reseña en inglés (3 a 5 oraciones)", height=180)
    if st.button("Clasificar con la red"):
        if resena.strip():
            r = predecir_sentimiento(resena)
            st.metric("Predicción", r["etiqueta"])
            st.write(f"P(positivo) = **{r['p_positivo']}** · confianza = **{r['confianza']}**")
            st.progress(float(r["p_positivo"]))
        else:
            st.warning("Escribe una reseña.")

with col2:
    st.subheader("2. Chat con el agente")
    for rol, contenido, tools in st.session_state.historial:
        with st.chat_message(rol):
            st.write(contenido)
            if tools is not None:
                st.caption("🛠️ Herramientas: " + (", ".join(tools) if tools else "ninguna"))
    pregunta = st.chat_input("Escribe al agente (puedes pegar una reseña)")
    if pregunta:
        st.session_state.historial.append(("user", pregunta, None))
        res = agente.invoke({"messages": [("user", pregunta)]},
                            config={"configurable": {"thread_id": st.session_state.thread_id}})
        msgs = res["messages"]
        idx = max(i for i, m in enumerate(msgs) if m.type == "human")
        usadas = [tc["name"] for m in msgs[idx:] for tc in (getattr(m, "tool_calls", None) or [])]
        st.session_state.historial.append(("assistant", texto(msgs[-1]), usadas))
        st.rerun()