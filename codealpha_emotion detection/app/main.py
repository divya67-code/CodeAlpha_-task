import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sys
import os

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from preprocess import load_audio, trim_silence
    from features import extract_all_features
    from tensorflow.keras.models import load_model    
    # Needs custom object scope if training script used Custom Attention Layer natively
    from model import AttentionLayer
    import tensorflow as tf
except ImportError as e:
    st.error(f"Error importing modules: {e}")

st.set_page_config(page_title="Deep Neural Emotion Intelligence", page_icon="🧠", layout="wide")

# Styling
st.markdown("""
<style>
.main {
    background-color: #0d1117;
    color: #c9d1d9;
}
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
}
h1 {
    color: #58a6ff;
}
.emotion-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #161b22;
    border: 1px solid #30363d;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Deep Neural Emotion Intelligence")
st.subheader("Advanced Speech-Based Human Emotion Recognition System")
st.markdown("Upload an audio file to predict the underlying emotion using a **Hybrid CNN + BiLSTM + Attention** model.")

@st.cache_resource
def load_ser_model():
    model_path = os.path.join("models", "final_model.h5")
    if os.path.exists(model_path):
        # Allow loading of custom Layer
        return load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    return None

@st.cache_data
def load_classes():
    class_path = os.path.join("models", "label_encoder_classes.npy")
    if os.path.exists(class_path):
        return np.load(class_path, allow_pickle=True)
    # Default RAVDESS + TESS generalized fallback
    return np.array(['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])

model = load_ser_model()
classes = load_classes()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1. Audio Input")
    
    # Try using the new audio_input for real-time recording if available, else standard file upload
    audio_source = st.radio("Choose Input Method", ["File Upload", "Record Microphone"])
    uploaded_file = None
    
    if audio_source == "File Upload":
        uploaded_file = st.file_uploader("Upload an audio file (.wav format)", type=["wav"])
    else:
        # Fallback for Streamlit versions supporting audio_input
        try:
            uploaded_file = st.audio_input("Record your voice")
        except AttributeError:
            st.warning("Your Streamlit version doesn't support built-in microphone recording yet. Please use File Upload or upgrade Streamlit.")
            uploaded_file = None
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner("Processing audio..."):
            # Save temporary file
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            audio, sr = load_audio("temp.wav")
            audio = trim_silence(audio)
            
            # Display Waveform
            st.markdown("### Waveform:")
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')
            ax.plot(audio, color='#58a6ff')
            ax.axis('off')
            st.pyplot(fig)
            
            features = extract_all_features(audio, sr)
            features = np.expand_dims(features, axis=0) # shape (1, features)
            features = np.expand_dims(features, axis=-1) # shape (1, features, 1)

with col2:
    if uploaded_file is not None:
        st.markdown("### 2. Emotion Prediction")
        if st.button("Predict Emotion"):
            if model is not None:
                with st.spinner("Running Inference..."):
                    predictions = model.predict(features)[0]
                    predicted_idx = np.argmax(predictions)
                    predicted_emotion = classes[predicted_idx]
                    intensity = predictions[predicted_idx]
                    
                    st.markdown(f"""
                    <div class="emotion-box">
                        <h2 style="color:white">Predicted Emotion: <span style="color:#58a6ff">{predicted_emotion.upper()}</span></h2>
                        <h4 style="color:gray">Confidence / Intensity Score: {intensity:.2f} / 1.0</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### Probability Distribution")
                    st.bar_chart(dict(zip(classes, predictions)))
            else:
                st.warning("⚠️ No trained model found in `models/final_model.h5`. Showing mock results for UI demonstration.")
                # Mock result for demo when model hasn't been trained yet
                mock_predictions = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
                predicted_idx = np.argmax(mock_predictions)
                predicted_emotion = classes[predicted_idx]
                intensity = mock_predictions[predicted_idx]
                
                st.markdown(f"""
                <div class="emotion-box">
                    <h2 style="color:white">Predicted Emotion: <span style="color:#58a6ff">{predicted_emotion.upper()}</span></h2>
                    <h4 style="color:gray">Confidence / Intensity Score: {intensity:.2f} / 1.0</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Probability Distribution")
                st.bar_chart(dict(zip(classes, mock_predictions)))

    else:
        st.info("Please upload an audio file to see the prediction.")
