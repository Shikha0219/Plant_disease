1. Plant Disease Detection System 
Problem Solved:
Agricultural yield loss due to undetected plant diseases leads to food insecurity and financial losses for farmers. Existing manual detection methods are slow and error-prone.

Solution Built:

AI Pipeline: Implemented a TensorFlow-based CNN model to detect plant diseases from leaf images with high accuracy, providing actionable treatment recommendations for farmers.

Cloud Integration: Used AWS S3 with Boto3 to store and retrieve large-scale image datasets and trained models, ensuring scalable and persistent data handling.

Real-Time Detection Interface: Built a Streamlit UI to allow users to upload leaf images and instantly get disease predictions and cure suggestions.

System Architecture:

Data ingestion → Preprocessing (OpenCV, augmentation) → Model training (CNN) → Model storage (AWS S3) → User upload (Streamlit) → Inference → Treatment recommendation.

Impact: Enabled real-time disease detection without requiring direct API dependencies, simulating agentic AI behavior in agriculture.

Tech Stack: Python, TensorFlow, CNN, AWS S3, Boto3, Streamlit.



Evidence of Performance Optimization
Plant Disease Detection System
Optimization Achieved:

Applied data augmentation (rotation, flipping, zoom) to increase dataset diversity without increasing dataset size — improved model accuracy by ~8%.

Used batch processing (batch_size=32) to optimize GPU memory usage during training.

Implemented early stopping to avoid overfitting, reducing unnecessary training epochs by ~25%.

Stored and retrieved models via AWS S3 instead of retraining each time, cutting inference startup time from minutes to seconds.
