import streamlit as st
import boto3
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import time

# Set page configuration with a custom theme
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #4CAF50;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description with better formatting
st.title("🌿 Plant Disease Detection System")
st.markdown("""
<div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
    <h3 style='color: #2e7d32;'>Welcome to Smart Plant Disease Detection!</h3>
    <p style='font-size: 1.1em; color: #333;'>
        This AI-powered system helps you identify plant diseases quickly and accurately. 
        Simply upload an image of a plant leaf, and our advanced machine learning model will:
        <ul>
            <li>Analyze the leaf characteristics</li>
            <li>Detect potential diseases</li>
            <li>Provide disease identification results</li>
        </ul>
    </p>
</div>
""", unsafe_allow_html=True)

# AWS S3 Configuration
ACCESS_KEY = "ACCESS_KEY"
SECRET_KEY = "SECRET_KEY"
BUCKET_NAME = 'plant-disease-detection-upload'
FOLDER_NAME = 'PlantVillage/'

# Class mapping with descriptions
CLASS_MAPPING = {
    'Pepper__bell___Bacterial_spot': {
        'name': 'Bell Pepper - Bacterial Spot',
        'description': 'A bacterial disease causing dark, raised spots on pepper leaves and fruits.'
    },
    'Pepper__bell___healthy': {
        'name': 'Bell Pepper - Healthy',
        'description': 'Healthy bell pepper plant with no signs of disease.'
    },
    'Potato___Early_blight': {
        'name': 'Potato - Early Blight',
        'description': 'A fungal disease causing dark spots with concentric rings on potato leaves.'
    },
    'Potato___healthy': {
        'name': 'Potato - Healthy',
        'description': 'Healthy potato plant with no signs of disease.'
    },
    'Potato___Late_blight': {
        'name': 'Potato - Late Blight',
        'description': 'A serious fungal disease causing dark, water-soaked spots on leaves.'
    },
    'Tomato__Tomato_mosaic_virus': {
        'name': 'Tomato - Mosaic Virus',
        'description': 'A viral disease causing mottled and distorted leaves.'
    },
    'Tomato_Bacterial_spot': {
        'name': 'Tomato - Bacterial Spot',
        'description': 'Bacterial disease causing small, dark spots on leaves and fruits.'
    },
    'Tomato_healthy': {
        'name': 'Tomato - Healthy',
        'description': 'Healthy tomato plant with no signs of disease.'
    }
}

# Initialize S3 Client with error handling
@st.cache_resource
def get_s3_client():
    try:
        return boto3.client('s3', 
                          aws_access_key_id=ACCESS_KEY, 
                          aws_secret_access_key=SECRET_KEY)
    except Exception as e:
        st.error(f"Failed to connect to AWS S3: {str(e)}")
        return None

s3 = get_s3_client()

@st.cache_data
def download_images_from_s3(bucket_name, folder_name):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        images, labels = [], []
        class_counts = {}
        
        # Get total number of files
        paginator = s3.get_paginator('list_objects_v2')
        total_files = 0
        
        # First, count total files
        for page in paginator.paginate(Bucket=bucket_name, Prefix=folder_name):
            for item in page.get('Contents', []):
                if not item['Key'].endswith('/'):
                    total_files += 1
        
        st.info(f"Found {total_files} images in the S3 bucket")
        
        # Now process all files
        processed_files = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=folder_name):
            for item in page.get('Contents', []):
                key = item['Key']
                if key.endswith('/'):
                    continue

                parts = key.split('/')
                if len(parts) < 3:
                    continue

                folder_name = parts[1]
                if folder_name in CLASS_MAPPING:
                    label = folder_name
                    class_counts[label] = class_counts.get(label, 0) + 1
                    temp_file = f"temp_{processed_files}.jpg"

                    try:
                        s3.download_file(bucket_name, key, temp_file)
                        image = cv2.imread(temp_file)
                        
                        if image is not None:
                            image = cv2.resize(image, (128, 128))
                            images.append(image)
                            labels.append(label)

                        os.remove(temp_file)
                        
                        processed_files += 1
                        progress = processed_files / total_files
                        progress_bar.progress(progress)
                        
                        # Update status every 100 files
                        if processed_files % 100 == 0:
                            status_text.text(f"Processing: {processed_files}/{total_files} images")

                    except Exception as e:
                        st.warning(f"⚠️ Skipped file {key}: {str(e)}")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

        progress_bar.progress(1.0)
        status_text.text("✅ Download completed!")
        
        # Display class distribution
        st.write("### Dataset Distribution")
        for class_name, count in class_counts.items():
            st.write(f"- {CLASS_MAPPING[class_name]['name']}: {count} images")
        
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return np.array(images), np.array(labels)

    except Exception as e:
        st.error(f"❌ Error downloading images: {str(e)}")
        return None, None

def train_model(X_train, y_train, X_test, y_test):
    progress_placeholder = st.empty()
    
    # Adjust model architecture based on dataset size
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASS_MAPPING), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Enhanced data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Custom callback for progress tracking
    class TrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / 20
            progress_placeholder.progress(progress)
            metrics_text = f"""
            Epoch {epoch + 1}/20
            Loss: {logs['loss']:.4f}
            Accuracy: {logs['accuracy']:.4f}
            Validation Loss: {logs['val_loss']:.4f}
            Validation Accuracy: {logs['val_accuracy']:.4f}
            """
            st.text(metrics_text)

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=20,
        callbacks=[TrainingCallback(), early_stopping]
    )

    # Save model locally first
    model_path = "plant_disease_model.h5"
    model.save(model_path)

    # Upload to S3
    try:
        s3.upload_file(model_path, BUCKET_NAME, f"models/{model_path}")
        st.success("✅ Model saved and uploaded to S3 successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to upload model to S3: {str(e)}")

    return model, history

def load_model_from_s3():
    try:
        model_path = "plant_disease_model.h5"
        s3.download_file(BUCKET_NAME, f"models/{model_path}", model_path)
        return load_model(model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

def main():
    st.sidebar.title("🔧 Control Panel")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["🔍 Disease Detection", "🤖 Train Model"])

    if page == "🤖 Train Model":
        st.header("Model Training Dashboard")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🚀 Start Training", key="train_button"):
                with st.spinner("Downloading training data..."):
                    images, labels = download_images_from_s3(BUCKET_NAME, FOLDER_NAME)

                if images is not None and len(images) > 0:
                    st.info(f"📊 Total Images: {len(images)}")
                    
                    # Preprocess data
                    images = images / 255.0
                    label_encoder = LabelEncoder()
                    label_encoder.fit(list(CLASS_MAPPING.keys()))
                    encoded_labels = label_encoder.transform(labels)
                    categorical_labels = to_categorical(encoded_labels)

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        images, categorical_labels, test_size=0.2, random_state=42,
                        stratify=encoded_labels  # Ensure balanced split
                    )

                    st.info("🎯 Starting model training...")
                    model, history = train_model(X_train, y_train, X_test, y_test)
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history.history['accuracy'], label='Training')
                    ax1.plot(history.history['val_accuracy'], label='Validation')
                    ax1.set_title('Model Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend()
                    
                    ax2.plot(history.history['loss'], label='Training')
                    ax2.plot(history.history['val_loss'], label='Validation')
                    ax2.set_title('Model Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    
                    st.pyplot(fig)

    else:  # Disease Detection Page
        st.header("🔍 Plant Disease Detection")
        
        # Create three columns
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            uploaded_file = st.file_uploader(
                "Upload a leaf image",
                type=["jpg", "jpeg", "png"],
                help="Drag and drop or click to upload an image"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("🔍 Detect Disease", key="detect_button"):
                    with st.spinner("Analyzing image..."):
                        model = load_model_from_s3()
                        
                        if model:
                            # Preprocess image
                            img_array = np.array(image)
                            img_resized = cv2.resize(img_array, (128, 128))
                            img_normalized = img_resized / 255.0
                            img_batch = np.expand_dims(img_normalized, axis=0)

                            # Make prediction
                            prediction = model.predict(img_batch)
                            predicted_class = np.argmax(prediction)
                            confidence = float(prediction[0][predicted_class] * 100)

                            # Get class label
                            label_encoder = LabelEncoder()
                            label_encoder.fit(list(CLASS_MAPPING.keys()))
                            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                            
                            # Display results in a nice format
                            st.markdown("""
                            <div style='background-color: #f0f9f0; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                                <h3 style='color: #2e7d32; margin-top: 0;'>Detection Results</h3>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <div style='margin-left: 20px;'>
                                    <p><strong>Detected Condition:</strong> {CLASS_MAPPING[predicted_label]['name']}</p>
                                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                                    <p><strong>Description:</strong> {CLASS_MAPPING[predicted_label]['description']}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
