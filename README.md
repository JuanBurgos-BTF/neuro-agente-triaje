# Agente neuro-agéntico de triaje de reseñas

## 1. Identificación, ruta y propuesta de valor

**Integrantes:** Juan Sebastián Burgos, Estiben Yessid Jiménez
**Ruta elegida:** Ruta 3, PLN con Embeddings + Agente de Triaje Inteligente
**Dataset:** IMDB Reviews (20.000 reseñas de train, 5.000 de validación y 25.000 de test)

### ¿Por qué red neuronal + agente y no solo un prompt a un LLM?

La red neuronal no solamente permite identificar si una reseña es positiva o negativa,
sino que también nos muestra una probabilidad que es posible analizar y comprobar. Por
ejemplo, en nuestras pruebas se obtuvo 0.7734 en una reseña positiva y 99.31 % de
confianza en una negativa. Otra ventaja es que funciona de manera local, sin depender
de servicios externos ni consumir cuota de la API de Google.

Por otra parte, el agente interpreta las solicitudes del usuario y decide qué
herramientas utilizar. Esto se comprobó en el Caso 3, donde encadenó la red neuronal y
la política de escalamiento a partir de una sola petición. Además, en la prueba de
memoria recordó el nombre "Ana" usando su `thread_id`, sin mezclar esa información con
otro hilo.

### Responsabilidades

| Integrante | Componente desarrollado |
|---|---|
| Juan Burgos | Desarrollo del proyecto, entrenamiento y evaluación de la red neuronal, implementación del agente inteligente e integración de las herramientas |
| Estiben Jiménez | Pruebas y validación del funcionamiento, integración de componentes y documentación de la solución |
