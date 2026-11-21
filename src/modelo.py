
import numpy as np
import tensorflow as tf

_MODELO = None

def cargar_modelo(ruta="models/modelo_base.keras"):
    global _MODELO
    if _MODELO is None:
        _MODELO = tf.keras.models.load_model(ruta)
    return _MODELO

def predecir_sentimiento(texto: str) -> dict:
    m = cargar_modelo()
    p = float(m.predict(np.array([texto], dtype=object), verbose=0)[0][0])
    etiqueta = "POSITIVO" if p >= 0.5 else "NEGATIVO"
    confianza = p if p >= 0.5 else 1 - p
    return {"etiqueta": etiqueta, "p_positivo": round(p, 4), "confianza": round(confianza, 4)}
