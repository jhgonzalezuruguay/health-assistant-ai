import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("symptom_based_diagnosis_model.h5")

model = load_trained_model()

# Disease dictionary
disease_info = {
    1: {"name": "Malaria", "description": "Malaria is a mosquito-borne infectious disease caused by Plasmodium parasites. It leads to symptoms like fever, chills, and flu-like illness. It can be treated with antimalarial drugs like chloroquine."},
    2: {"name": "Dengue", "description": "Dengue is a viral illness transmitted by mosquitoes, characterized by high fever, severe headache, pain behind the eyes, joint pain, and rash. There's no specific treatment, but supportive care can help."},
    3: {"name": "Influenza", "description": "Influenza (Flu) is a viral respiratory infection that causes fever, body aches, sore throat, and fatigue. Vaccines are available to reduce the severity of symptoms."},
    4: {"name": "Tuberculosis", "description": "Tuberculosis (TB) is a bacterial infection that primarily affects the lungs. Symptoms include a persistent cough, weight loss, and night sweats. TB is treatable with antibiotics over an extended period."},
    5: {"name": "COVID-19", "description": "COVID-19 is a viral respiratory disease caused by the SARS-CoV-2 virus. Symptoms include fever, dry cough, and difficulty breathing. Vaccines and other treatments can help manage the disease."},
    6: {"name": "Cholera", "description": "Cholera is a waterborne bacterial infection causing severe diarrhea and dehydration. It can be treated with oral rehydration therapy and antibiotics."},
    7: {"name": "Hepatitis", "description": "Hepatitis is an inflammation of the liver, often caused by viral infections. Symptoms include jaundice, fatigue, and abdominal pain. Treatment varies depending on the type of hepatitis."},
    8: {"name": "Typhoid", "description": "Typhoid fever is a bacterial infection caused by Salmonella typhi. It presents with high fever, abdominal pain, and weakness. Treatment involves antibiotics."},
    9: {"name": "Chickenpox", "description": "Chickenpox is a viral infection caused by the varicella-zoster virus. It results in an itchy rash, fever, and tiredness. Treatment includes antiviral drugs and antihistamines."},
    10: {"name": "Measles", "description": "Measles is a highly contagious viral infection that causes fever, cough, runny nose, and a red rash. Vaccination can prevent this disease."},
    11: {"name": "Pneumonia", "description": "Pneumonia is an infection that causes inflammation of the lungs, leading to cough, fever, and chest pain. It can be caused by bacteria, viruses, or fungi and is treated with antibiotics or antivirals."},
    12: {"name": "Asthma", "description": "Asthma is a chronic respiratory disease that causes difficulty breathing due to inflammation and narrowing of the airways. Symptoms include wheezing, coughing, and shortness of breath. Inhalers and medications help control it."},
    13: {"name": "Cancer", "description": "Cancer refers to a group of diseases involving abnormal cell growth that can invade other parts of the body. Symptoms vary depending on the type, and treatment includes surgery, chemotherapy, and radiation."},
    14: {"name": "HIV/AIDS", "description": "HIV is a virus that weakens the immune system by destroying CD4 cells. If untreated, it leads to AIDS. Treatment with antiretroviral drugs (ART) can control the virus and prevent progression."},
    15: {"name": "Diabetes", "description": "Diabetes is a metabolic disease characterized by high blood sugar. There are two main types: Type 1 (insulin-dependent) and Type 2 (insulin resistance). It can be managed with medication, lifestyle changes, and insulin therapy."},
    16: {"name": "Hypertension", "description": "Hypertension (high blood pressure) is a condition where the force of blood against the artery walls is too high. It can be managed with medications and lifestyle changes."},
    17: {"name": "Anemia", "description": "Anemia is a condition where you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. Symptoms include fatigue and weakness. It can be treated with iron supplements and dietary changes."},
    18: {"name": "Migraine", "description": "Migraine is a neurological condition characterized by severe, recurring headaches often accompanied by nausea, vomiting, and sensitivity to light. Pain relief and preventative treatments are available."},
    19: {"name": "Arthritis", "description": "Arthritis is inflammation of the joints, causing pain, swelling, and stiffness. The two most common types are osteoarthritis and rheumatoid arthritis. Treatment includes pain management and physical therapy."},
    20: {"name": "Stroke", "description": "A stroke occurs when the blood supply to part of the brain is interrupted. Symptoms include sudden numbness, confusion, difficulty speaking, and vision problems. Immediate medical attention is essential."},
    21: {"name": "Epilepsy", "description": "Epilepsy is a neurological disorder marked by recurrent seizures. Seizures can be controlled with medications, and in some cases, surgery is required."},
    22: {"name": "Alzheimer's Disease", "description": "Alzheimer's is a progressive neurological disease that causes memory loss, confusion, and changes in behavior. There is no cure, but medications can manage symptoms."},
    23: {"name": "Pneumothorax", "description": "Pneumothorax is a condition where air leaks into the space between the lungs and chest wall, causing the lung to collapse. It can result in sudden chest pain and difficulty breathing."},
    24: {"name": "Cystic Fibrosis", "description": "Cystic fibrosis is a genetic disorder that affects the lungs and digestive system, causing difficulty breathing and poor growth. Treatment includes medications, lung therapy, and dietary support."},
    25: {"name": "Gastritis", "description": "Gastritis is inflammation of the stomach lining, often caused by infection, alcohol use, or long-term use of nonsteroidal anti-inflammatory drugs (NSAIDs). Treatment involves antacids or antibiotics."},
    26: {"name": "Ulcerative Colitis", "description": "Ulcerative colitis is an inflammatory bowel disease causing long-lasting inflammation and ulcers in the colon and rectum. Treatment includes medications and sometimes surgery."},
    27: {"name": "Multiple Sclerosis", "description": "Multiple sclerosis is a disease in which the immune system attacks the protective covering of nerve fibers, causing symptoms like fatigue, difficulty walking, and numbness."},
    28: {"name": "Parkinson's Disease", "description": "Parkinson's disease is a neurodegenerative disorder that causes tremors, stiffness, and difficulty with movement. Treatment focuses on managing symptoms, typically with medications."},
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
st.subheader("¿Como te sientes?!")
st.write("Selecciona como te sientes")

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
        disease_info = disease_info.get(predicted_class + 1, {"name": "Consulte a un Profesional en Salud Mental", "description": "Consulte a un Profesional en Salud Mental"})

        # Show prediction result
        st.success(f"Predicted Disease: {disease_info['name']}")

        # Show disease description
        st.write(f"**Description:** {disease_info['description']}")

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
