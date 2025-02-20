import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("symptom_based_diagnosis_model.h5")

model = load_trained_model()

# Disease dictionary with descriptions
disease_info = {
    1: {"name": "Ansiedad", "description": "Un estado de inquietud, temor o preocupación persistente."},
    2: {"name": "Depresión", "description": "Un trastorno que causa sentimientos persistentes de tristeza y pérdida de interés."},
    3: {"name": "Pánico", "description": "Ataques de miedo intenso y repentino con síntomas físicos."},
    4: {"name": "Esquizofrenia", "description": "Un trastorno que afecta la capacidad de pensar, sentir y comportarse con claridad."},
    5: {"name": "Trastorno Bipolar", "description": "Condición caracterizada por cambios extremos de ánimo, incluyendo episodios maníacos y depresivos."},
    6: {"name": "Obsesivo-Compulsivo", "description": "Trastorno en el cual las personas tienen pensamientos (obsesiones) y comportamientos (compulsiones) repetitivos."},
    7: {"name": "Estrés", "description": "Reacción del cuerpo a situaciones de presión o demanda."},
    8: {"name": "Trastornos de Alimentación", "description": "Condiciones serias relacionadas con comportamientos alimentarios que afectan negativamente la salud y las emociones."},
    9: {"name": "Disfunción Sexual", "description": "Problemas que impiden una satisfacción plena durante la actividad sexual."},
    10: {"name": "Adicción", "description": "Trastorno en el que una persona no puede dejar de usar una sustancia o realizar una actividad."},
    11: {"name": "Parafilasis", "description": "Trastornos de interés sexual atípico."},
    12: {"name": "Trastorno de Personalidad", "description": "Patrones duraderos de comportamiento y experiencia que se desvían de las expectativas de la cultura del individuo."},
    13: {"name": "Trastorno de Excreción", "description": "Condiciones donde una persona tiene problemas con el control de la eliminación de orina o heces."},
    14: {"name": "Déficit atencional", "description": "Trastorno caracterizado por problemas de atención, hiperactividad e impulsividad."},
    15: {"name": "Trastornos sicosomáticos", "description": "Trastornos físicos que tienen un origen emocional o mental."},
    16: {"name": "Trastornos del Lenguaje", "description": "Problemas que afectan la capacidad de una persona para entender y producir lenguaje."},
}

# Symptom to index mapping
symptom_to_index = {
    "cansancio": 0,
    "ansiedad": 1,
    "irritable": 2,
    "tristeza": 3,
    "nausea": 4,
    "frustración": 5,
    "desinterés": 6,
    "insomnio": 7,
    "desconcentrado": 8,
    "desesperanza": 9
}

# Page title
st.title("VITAL: Asistente de salud mental con IA")

# Input form with checkboxes or radio buttons
st.subheader("¿Cómo te sientes?")
st.write("Selecciona cómo te sientes")

# Symptom options
symptom_options = list(symptom_to_index.keys())

# User selects symptoms
selected_symptoms = st.multiselect("Selecciona los Síntomas", symptom_options)

# Predict button
if st.button("Diagnóstico"):
    if selected_symptoms:
        # Create a 142-length feature vector, where each symptom corresponds to a specific index
        features = np.zeros(142)  # Assuming model expects 142 features

        # Encoding logic: set corresponding indices to 1 based on selected symptoms
        for symptom in selected_symptoms:
            index = symptom_to_index.get(symptom.strip())  # Map symptom to its index
            if index is not None:
                features[index] = 1

        # Reshape the features to match the expected input shape of the model
        features = features.reshape(1, -1)  # Shape should be (1, 142)

        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)

        # Retrieve disease information using the predicted class
        disease = disease_info.get(predicted_class + 1, {"name": "Consulte a un Profesional en Salud Mental", "description": "Consulte a un Profesional en Salud Mental"})

        # Show prediction result
        st.success(f"Predicted Disease: {disease['name']}")

        # Show disease description
        st.write(f"**Descripción:** {disease['description']}")

        # Confidence level message
        prediction_prob = np.max(prediction)
        if prediction_prob >= 0.8:
            confidence_message = "El modelo es altamente confiable en la predicción."
        elif prediction_prob >= 0.5:
            confidence_message = "El modelo es confiable en la predicción."
        else:
            confidence_message = "El modelo es moderadamente confiable en la predicción."

        # Display confidence message
        st.write(f"Confidence: {prediction_prob * 100:.2f}%")
        st.write(confidence_message)

    else:
        st.warning("Selecciona un síntoma para su diagnóstico!")

# Footer
st.write("VITAL LE AGRADECE POR CONFIAR Y USAR NUESTRO SERVICIO!! ❤️")
