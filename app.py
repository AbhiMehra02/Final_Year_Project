import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import mode

# Force TensorFlow to use NHWC format
tf.keras.backend.set_image_data_format('channels_last')

# Class index to name mapping
CLASS_MAPPING = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}


# Model nomenclature
MODEL_NAMING_CONVENTIONS = """
**Model Nomenclature Reference**:
- **IRV2**: InceptionResNetV2  
- **Xptn**: Xception  
- **wa**: Without augmentation  
- **aug**: Augmentation applied  
- **fl**: Fine-tune last layer  
- **fe**: Feature extraction  
- **fwm**: Fine-tune whole model  
- **3,7**: Patience in model training
"""


# Helper functions
def preprocess_image(image, target_size=(224, 224)):
    """Resize and normalize the image to match model input requirements."""
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)  # Add batch dimension

def load_models(folder_path):
    """Load all models from the specified folder."""
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.keras')]
    models = {}
    for file in model_files:
        model_name = os.path.splitext(file)[0]
        model_path = os.path.join(folder_path, file)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            models[model_name] = model
        except Exception as e:
            st.warning(f"Error loading model {file}: {e}")
    return models

def predict(model, image):
    """Generate predictions for the input image."""
    predictions = model.predict(image)[0]  # Predict and remove batch dimension
    return predictions

def plot_predictions(predictions, class_mapping):
    """Plot histogram of predictions."""
    classes = list(class_mapping.values())
    confidence = predictions
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(classes, confidence, color='skyblue')
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence for Each Class")
    plt.tight_layout()
    return fig

def ensemble_predict(models, image, method='avg'):
    """Generate ensemble predictions using average or voting."""
    all_predictions = []
    
    # Collect predictions from each model
    for model in models:
        all_predictions.append(predict(model, image))
    
    # Convert list to a numpy array for easier manipulation
    all_predictions = np.array(all_predictions)
    
    if method == 'avg':
        # Average the predictions across all models
        avg_predictions = all_predictions.mean(axis=0)
        return avg_predictions
    
    elif method == 'voting':
        # Majority voting (find the most frequent class in each prediction)
        votes = mode(all_predictions, axis=0).mode[0]
        return votes
    else:
        raise ValueError("Invalid ensemble method selected. Use 'avg' or 'voting'.")

# Streamlit App
st.title("Multi-Model Image Classifier with Ensemble Learning")
st.sidebar.markdown(MODEL_NAMING_CONVENTIONS)

# Sidebar for inputs
with st.sidebar:
    st.header("Input Options")
    
    # Load models
    models_folder = "models"
    models = load_models(models_folder)
    if not models:
        st.error("No models were loaded. Please check the models folder.")
        st.stop()

    model_names = list(models.keys())
    
    # Allow user to choose between single or multiple model prediction
    prediction_mode = st.selectbox("Choose Prediction Mode", ("Single Model", "Multiple Models"))
    
    if prediction_mode == "Single Model":
        selected_model_name = st.selectbox("Choose a model:", model_names)
        selected_model = models[selected_model_name]
    elif prediction_mode == "Multiple Models":
        selected_models = st.multiselect("Select models for ensemble prediction:", model_names)
        selected_models = [models[model] for model in selected_models]
        
        # Choose ensemble method (Voting or Averaging)
        ensemble_method = st.selectbox("Choose Ensemble Method", ("avg", "voting"))
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# Main section
if uploaded_file:
    # Create a two-column layout
    col1, col2 = st.columns([1, 2])  # Adjust the column widths as necessary

    # Left column: Uploaded Image
    with col1:
        st.markdown("### Uploaded Image:")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_image = preprocess_image(image)

    # Right column: Predictions and Histogram
    with col2:
        st.markdown("### Predictions and Confidence Histogram")
        try:
            if prediction_mode == "Single Model":
                # Single model prediction
                predictions = predict(selected_model, input_image)
                
                # Map predictions to class names and confidence scores
                top_indices = predictions.argsort()[-3:][::-1]  # Top 3 indices
                st.markdown("### Top Predictions:")
                for idx in top_indices:
                    class_name = CLASS_MAPPING[idx]
                    confidence = predictions[idx]
                    st.write(f"- **{class_name}**: {confidence:.2%}")

                # Plot predictions as a histogram
                hist_fig = plot_predictions(predictions, CLASS_MAPPING)
                st.pyplot(hist_fig)
            
            elif prediction_mode == "Multiple Models":
                # Ensemble prediction
                ensemble_predictions = ensemble_predict(selected_models, input_image, method=ensemble_method)
                
                # Map predictions to class names and confidence scores
                top_indices = ensemble_predictions.argsort()[-3:][::-1]  # Top 3 indices
                st.markdown("### Top Predictions from Ensemble:")
                for idx in top_indices:
                    class_name = CLASS_MAPPING[idx]
                    confidence = ensemble_predictions[idx]
                    st.write(f"- **{class_name}**: {confidence:.2%}")

                # Plot ensemble predictions as a histogram
                hist_fig = plot_predictions(ensemble_predictions, CLASS_MAPPING)
                st.pyplot(hist_fig)
            
        except Exception as e:
            st.error(f"Error processing image or predicting: {e}")
