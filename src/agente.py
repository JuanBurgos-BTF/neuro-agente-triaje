
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from src.herramientas import clasificar_sentimiento, consultar_politica_escalamiento

SYSTEM_PROMPT = (
    "Eres un agente de triaje de resenas y tickets. Responde en espanol y de forma breve. "
    "Si el usuario solo saluda o hace una pregunta general, responde sin usar herramientas. "
    "Si entrega una resena para analizar, usa clasificar_sentimiento. "
    "Usa consultar_politica_escalamiento SOLO si el usuario pide expresamente una accion, prioridad o escalamiento. Si solo pide clasificar, responde unicamente con la clasificacion."
)

def crear_agente(modelo_llm=None):
    llm = ChatGoogleGenerativeAI(
        model=modelo_llm or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0,
        max_output_tokens=512,
    )
    return create_agent(
        llm,
        tools=[clasificar_sentimiento, consultar_politica_escalamiento],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
