
import os
import joblib
import numpy as np
from PIL import Image
import cv2




def preprocess_image(image, target_size=(112, 92)):
    if isinstance(image, str):
        image = Image.open(image)
    if image.mode != 'L':
        image = image.convert('L')
    if image.size != target_size[::-1]:
        image = image.resize(target_size[::-1], Image.Resampling.LANCZOS)
    image_array = np.array(image)
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    return image



def predict_multiple_identities(image, model_components):
    try:
        # Unpack model components
        pca = model_components['pca']
        scaler = model_components['scaler']
        classifier = model_components['classifier']
        subject_dict = model_components['subject_dict']
        
        # Load HaarCascade face detector
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Load image using OpenCV for face detection
        img_cv = image
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect all faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If no faces detected
        if len(faces) == 0:
            print(f"No faces detected ")
            return []
            
        # Create a copy of the image for drawing results
        img_with_results = img_cv.copy()
        
        # List to store results for each face
        results = []
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Add margin around the face (20% of face size)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            # Ensure coordinates are within image bounds
            x_start = max(0, x - margin_x)
            y_start = max(0, y - margin_y)
            x_end = min(img_cv.shape[1], x + w + margin_x)
            y_end = min(img_cv.shape[0], y + h + margin_y)
            
            # Crop the face region
            face_region = gray[y_start:y_end, x_start:x_end]
            face_pil = Image.fromarray(face_region)
            
            # Preprocess the detected face
            img_preprocessed = preprocess_image(face_pil, target_size=(112, 92))
            
            # Convert to array and flatten
            img_array = np.array(img_preprocessed).flatten().reshape(1, -1)
            
            # Apply transformations
            img_scaled = scaler.transform(img_array)
            img_pca = pca.transform(img_scaled)
            
            # Predict and get confidence
            prediction = classifier.predict(img_pca)[0]
            confidence = classifier.decision_function(img_pca).max()
            
            # Get subject name
            subject_name = subject_dict.get(prediction, f"Subject {prediction+1}")
            
            # Store result for this face
            face_result = {
                'face_id': i,
                'subject_id': int(prediction),
                'subject_name': subject_name,
                'confidence': float(confidence),
                'position': (x, y, w, h)
            }
            
            results.append(face_result)
            
            # Draw rectangle around face (green for high confidence, red for low)
            color = (0, 255, 0) if confidence > 1.0 else (0, 0, 255)
            cv2.rectangle(img_with_results, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label text
            label = f"{subject_name} ({confidence:.2f})"
            
            font_scale = w / 200  # Adjust this divisor to change the relative size
            cv2.putText(img_with_results, 
                        subject_name, 
                        (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale,  # Dynamic font scale based on face width
                        (255, 255, 255),  # White color
                        1)  
        
        return results , img_with_results
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_model(model_dir = 'model/updated_model'):
    pca = joblib.load(os.path.join(model_dir, 'pca_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    classifier = joblib.load(os.path.join(model_dir, 'svm_classifier.pkl'))
    subject_dict = joblib.load(os.path.join(model_dir, 'subject_names.pkl'))
    return {
        'pca': pca,
        'scaler': scaler,
        'classifier': classifier,
        'subject_dict': subject_dict
    }