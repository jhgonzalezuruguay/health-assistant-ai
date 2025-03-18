import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado (asegúrate de que se haya reentrenado con el nuevo mapeo de síntomas)
#@st.cache_resource
#def load_trained_model():
#    return load_model("symptom_based_diagnosis_model_updated.h5")
#model = load_model(r"C:\Users\user\Desktop\stanford\Untitled1.ipynb0 - Colab_files\my_model.keras")
model = "my_model.keras"

# Diccionario actualizado de enfermedades mentales con descripciones
disease_info = {
    1: {"name": "Trastorno de Ansiedad Generalizada", "description": "Ansiedad excesiva y preocupación persistente en múltiples situaciones, interfiriendo con la vida diaria."},
    2: {"name": "Depresión Mayor", "description": "Estado de ánimo deprimido, pérdida de interés, cambios en el apetito y dificultades en el funcionamiento diario."},
    3: {"name": "Trastorno de Pánico", "description": "Ataques de pánico recurrentes, con síntomas físicos intensos como palpitaciones, sudoración y miedo intenso."},
    4: {"name": "Esquizofrenia", "description": "Trastorno severo que afecta el pensamiento, la percepción y la interacción social, pudiendo incluir alucinaciones y delirios."},
    5: {"name": "Trastorno Bipolar", "description": "Alternancia de episodios de manía y depresión, con cambios significativos en el estado de ánimo y energía."},
    6: {"name": "Trastorno Obsesivo-Compulsivo", "description": "Presencia de obsesiones y compulsiones que generan ansiedad y afectan la rutina diaria."},
    7: {"name": "Trastorno de Estrés Postraumático (TEPT)", "description": "Respuesta prolongada a eventos traumáticos con recuerdos intrusivos, pesadillas y evitación de estímulos relacionados."},
    8: {"name": "Trastornos de la Alimentación", "description": "Patrones alimenticios anormales, preocupación excesiva por la imagen corporal y comportamientos extremos relacionados con la comida."},
    9: {"name": "Disfunción Sexual", "description": "Problemas en la función sexual que afectan el deseo, la respuesta y el placer, pudiendo tener origen psicológico."},
    10: {"name": "Adicción", "description": "Dependencia a sustancias o comportamientos, con consecuencias negativas en el ámbito físico, emocional y social."},
    11: {"name": "Trastorno de la Personalidad", "description": "Patrones de pensamiento y comportamiento inflexibles y desadaptativos que afectan la forma de relacionarse con los demás."},
    12: {"name": "Trastorno de Déficit de Atención e Hiperactividad (TDAH)", "description": "Dificultades para mantener la atención, hiperactividad e impulsividad que interfieren en la vida diaria."},
    13: {"name": "Trastorno de Ansiedad Social", "description": "Miedo intenso a situaciones sociales que genera evitación y deterioro en la interacción interpersonal."},
    14: {"name": "Fobias Específicas", "description": "Miedo irracional y persistente hacia objetos o situaciones concretas, con respuestas de ansiedad desproporcionadas."},
    15: {"name": "Trastorno Disociativo", "description": "Alteraciones en la integración de la conciencia, memoria e identidad, a menudo vinculadas a experiencias traumáticas."},
    16: {"name": "Trastorno Límite de la Personalidad", "description": "Inestabilidad en las relaciones interpersonales, autoimagen y emociones, acompañada de impulsividad significativa."}
}

# Diccionario actualizado de síntomas a índices (asegúrate de que el modelo se haya entrenado con este mapeo)
symptom_to_index = {
    "cansancio": 0,
    "ansiedad": 1,
    "irritabilidad": 2,
    "tristeza": 3,
    "náusea": 4,
    "frustración": 5,
    "desinterés": 6,
    "insomnio": 7,
    "desconcentración": 8,
    "desesperanza": 9,
    "cambio de apetito": 10,
    "miedo irracional": 11,
    "ataques de pánico": 12,
    "baja autoestima": 13,
    "pensamientos negativos": 14,
    "apático": 15,
    "hipervigilancia": 16,
    "tensión muscular": 17,
    "agitación": 18
}

# Título de la aplicación
st.title("VITAL: Asistente de salud mental con IA")

# Formulario de entrada
st.subheader("¿Cómo te sientes?")
st.write("Selecciona los síntomas que presentas:")

# Opciones de síntomas basadas en el diccionario actualizado
symptom_options = list(symptom_to_index.keys())
selected_symptoms = st.multiselect("Síntomas", symptom_options)

# Botón para realizar la predicción
if st.button("Diagnóstico"):
    if not selected_symptoms:
        st.error("Por favor, selecciona al menos un síntoma para realizar el diagnóstico.")
    else:
        # Actualiza el vector de características a la longitud de los síntomas disponibles
        num_features = len(symptom_to_index)
        features = np.zeros(num_features)
        
        # Codificar los síntomas seleccionados
        for symptom in selected_symptoms:
            index = symptom_to_index.get(symptom.strip())
            if index is not None and index < num_features:
                features[index] = 1
        
        # Remodelar el vector para adaptarse a la entrada del modelo
        features = features.reshape(1, -1)
        
        # Realizar la predicción
        ######prediction = model.predict(features)
        ######predicted_class = np.argmax(prediction)
        #####prediction_prob = np.max(prediction)
        
        # Obtener la información de la enfermedad basada en la clase predicha
        disease = disease_info.get(predicted_class + 1, {
            "name": "Diagnóstico Inconcluso",
            "description": "Se recomienda consultar a un profesional en salud mental para una evaluación completa."
        })
        
        # Mostrar el resultado
        st.success(f"Diagnóstico: {disease['name']}")
        st.write(f"**Descripción:** {disease['description']}")
        st.write(f"**Confianza en la predicción:** {prediction_prob * 100:.2f}%")
        
        # Mensaje de confianza
        if prediction_prob >= 0.8:
            st.info("El modelo es altamente confiable en la predicción.")
        elif prediction_prob >= 0.5:
            st.info("El modelo es confiable en la predicción.")
        else:
            st.warning("El modelo es moderadamente confiable en la predicción. Se recomienda consultar a un profesional.")


        # Display confidence message
        st.write(f"Confidence: {prediction_prob * 100:.2f}%")
        st.write(confidence_message)
else:
        st.warning("Selecciona un síntoma para su diagnóstico!")

# Footer
st.write("VITAL LE AGRADECE POR CONFIAR Y USAR NUESTRO SERVICIO!! ❤️")
