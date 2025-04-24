import streamlit as st
import librosa
import numpy as np
import json
import pandas as pd
import joblib
from hive_health import compute_hive_health  # Import the function from hive_health.py

# Load the trained model from the .pkl file
model = joblib.load('random_forest_model_test.pkl')

# Function to convert numpy types to native Python types (recursively for nested structures)
def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Function to extract selected audio features
def extract_features(audio_file_path, sample_rate=22050):
    # Load the audio file
    y, sr = librosa.load(audio_file_path, sr=sample_rate)
    
    # Extract RMS Energy and compute mean
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    
    # Extract Spectral Centroid and compute mean
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = float(np.mean(spectral_centroid))
    
    # Extract Spectral Contrast and compute mean
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = float(np.mean(spectral_contrast))
    
    # Extract MFCC (1 to 13) and compute mean for each coefficient
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1).tolist()
    
    features = {
        'RMS_Energy': rms_mean,
        'Spectral_Centroid': spectral_centroid_mean,
        'Spectral_Contrast': spectral_contrast_mean,
        'MFCC_1': mfcc_means[0],
        'MFCC_2': mfcc_means[1],
        'MFCC_3': mfcc_means[2],
        'MFCC_4': mfcc_means[3],
        'MFCC_5': mfcc_means[4],
        'MFCC_6': mfcc_means[5],
        'MFCC_7': mfcc_means[6],
        'MFCC_8': mfcc_means[7],
        'MFCC_9': mfcc_means[8],
        'MFCC_10': mfcc_means[9],
        'MFCC_11': mfcc_means[10],
        'MFCC_12': mfcc_means[11],
        'MFCC_13': mfcc_means[12],
    }
    return features

def main():
    st.title("Audio Feature Extraction")

    uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "flac"])
    
    st.sidebar.header("Additional Inputs")
    hive_pressure = st.sidebar.number_input("Hive Pressure (Pa)", min_value=0.0, format="%.2f", value=0.0)
    hive_temperature = st.sidebar.number_input("Hive Temperature (°C)", min_value=-50.0, max_value=100.0, format="%.2f", value=25.0)
    hive_humidity = st.sidebar.number_input("Hive Humidity (%)", min_value=0.0, max_value=100.0, format="%.2f", value=50.0)
    weather_temperature = st.sidebar.number_input("Weather Temperature (°C)", min_value=-50.0, max_value=100.0, format="%.2f", value=20.0)
    weather_pressure = st.sidebar.number_input("Weather Pressure (Pa)", min_value=0.0, format="%.2f", value=101325.0)
    weather_humidity = st.sidebar.number_input("Weather Humidity (%)", min_value=0.0, max_value=100.0, format="%.2f", value=60.0)
    
    additional_data = {
        "hive_pressure": hive_pressure,
        "hive_temperature": hive_temperature,
        "hive_humidity": hive_humidity,
        "weather_temperature": weather_temperature,
        "weather_pressure": weather_pressure,
        "weather_humidity": weather_humidity
    }

    if uploaded_file is not None:
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file, format="audio/wav")
        
        features = extract_features("temp_audio_file")
        
        all_data = {**features, **additional_data}
        
        st.subheader("Extracted Features")
        st.write("### RMS Energy (mean):", features['RMS_Energy'])
        st.write("### Spectral Centroid (mean):", features['Spectral_Centroid'])
        st.write("### Spectral Contrast (mean):", features['Spectral_Contrast'])
        st.write("### MFCC Features (mean of coefficients 1 to 13):")
        st.write(pd.DataFrame([features[f'MFCC_{i}'] for i in range(1, 14)]).T)
        
        st.write("### Additional Inputs")
        for key, value in additional_data.items():
            st.write(f"{key.replace('_', ' ').title()}: {value}")
        
        with open('extracted_features.json', 'w') as json_file:
            json.dump(all_data, json_file, indent=4)
        
        st.success("Features and additional data saved to 'extracted_features.json'")

        # Load the extracted features from JSON
        with open('extracted_features.json', 'r') as f:
            extracted_data = json.load(f)
        
        # Compute Hive Health Score and Category using the external function
        hive_health_score, health_category = compute_hive_health(extracted_data)
        
        # Show the Hive Health Score and Category
        st.subheader("Hive Health Score")
        st.write(f"Score: {hive_health_score:.2f}%")
        st.write(f"Category: {health_category}")

        # Prepare input for the model
        input_features = [
            extracted_data['hive_temperature'],
            extracted_data['hive_humidity'],
            extracted_data['hive_pressure'],
            extracted_data['weather_temperature'],
            extracted_data['weather_humidity'],
            extracted_data['weather_pressure'],
            extracted_data['RMS_Energy'],
            extracted_data['Spectral_Centroid'],
            extracted_data['Spectral_Contrast'],
            # Add all MFCC features
            extracted_data['MFCC_1'], extracted_data['MFCC_2'], extracted_data['MFCC_3'],
            extracted_data['MFCC_4'], extracted_data['MFCC_5'], extracted_data['MFCC_6'],
            extracted_data['MFCC_7'], extracted_data['MFCC_8'], extracted_data['MFCC_9'],
            extracted_data['MFCC_10'], extracted_data['MFCC_11'], extracted_data['MFCC_12'], extracted_data['MFCC_13']
        ]
        input_features = np.array(input_features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_features)
        
        st.subheader("Prediction Result")
        st.write(f"Predicted Class: {prediction[0]}")
        
        # Prepare a new dictionary for the results file
        results_data = {
            "hive_health_score": hive_health_score,
            "health_category": health_category,
            "predicted_class": prediction[0]
        }
        
        # Convert all numpy types to native Python types before saving to JSON
        results_data = convert_numpy_types(results_data)
        
        # Save the Hive Health and Prediction Results to a separate JSON file
        with open('prediction_results.json', 'w') as results_file:
            json.dump(results_data, results_file, indent=4)
        
        st.success("Prediction results saved to 'prediction_results.json'")

if __name__ == "__main__":
    main()
