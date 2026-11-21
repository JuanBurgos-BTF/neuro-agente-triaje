
from langchain_core.tools import tool
from src.modelo import predecir_sentimiento

@tool
def clasificar_sentimiento(texto: str) -> str:
    """Clasifica con la red neuronal local (Embedding+Dense, entrenada en IMDB) el
    sentimiento de UNA reseña en INGLES. Usala solo cuando el usuario entregue el
    texto de una reseña o ticket para analizar. Devuelve etiqueta (POSITIVO/NEGATIVO)
    y la probabilidad. No sirve para saludos ni preguntas sobre politicas."""
    r = predecir_sentimiento(texto)
    return f"{r['etiqueta']} | P(positivo)={r['p_positivo']} | confianza={r['confianza']}"

@tool
def consultar_politica_escalamiento(etiqueta: str, confianza: float) -> str:
    """Aplica las reglas de escalamiento del equipo de soporte. Recibe la etiqueta
    (POSITIVO o NEGATIVO) y la confianza (0 a 1) de la clasificacion. Usala para
    decidir que accion tomar o cuando el usuario pregunte por las politicas de
    escalamiento. No clasifica texto."""
    etiqueta = etiqueta.upper().strip()
    if confianza < 0.60:
        return "ESCALAR A REVISION HUMANA: confianza menor a 0.60, el modelo no es fiable en este caso."
    if etiqueta == "NEGATIVO" and confianza >= 0.90:
        return "PRIORIDAD ALTA: responder en menos de 2 horas y notificar al supervisor."
    if etiqueta == "NEGATIVO":
        return "PRIORIDAD MEDIA: responder en 24 horas con el protocolo estandar de disculpa."
    return "PRIORIDAD BAJA: agradecer al cliente y archivar."
