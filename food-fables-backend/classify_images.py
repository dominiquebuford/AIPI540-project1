import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Assuming the model and class labels are constants for the application
MODEL_PATH = 'models/veg-classifier.h5'
CLASS_LABELS = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot',
                'Cailiflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

def load_model():
    """Load and return the TensorFlow model."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Cache the model upon first load to avoid reloading it for each request
model = load_model()

def preprocess_image(uploaded_file):
    """Process the uploaded file to prepare it for the model."""
    # Convert to a file-like object
    img = Image.open(BytesIO(uploaded_file.read()))
    img = img.resize((224, 224))  # Resizing to the target size
    img_array = np.array(img) / 255.0  # Convert to numpy array and scale
    img_batch = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_batch

def classify_image(img_batch):
    """Classify the image and return the class label."""
    predictions = model.predict(img_batch)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = CLASS_LABELS[predicted_class_idx]
    return predicted_class_label