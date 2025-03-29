import cv2
import numpy as np
import os

def load_known_faces():
    known_faces = []
    known_names = []
    known_face_images = []  # Store original face images for training
    
    # Create known_faces directory if it doesn't exist
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
        print("Created 'known_faces' directory. Please add photos of known people.")
        return known_faces, known_names, None

    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load known faces from directory
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read image
            image = cv2.imread(f'known_faces/{filename}')
            if image is None:
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try different scale factors to detect faces
            faces = []
            # Try with more sensitive parameters
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,  # Lower scale factor to detect more faces
                minNeighbors=3,   # Lower min neighbors for more detections
                minSize=(30, 30)  # Smaller minimum face size
            )
            
            if len(faces) > 0:
                # Take the first face found
                x, y, w, h = faces[0]
                face = gray[y:y+h, x:x+w]  # Use grayscale for LBPH
                
                # Resize face to standard size
                face = cv2.resize(face, (150, 150))
                
                # Apply histogram equalization to improve recognition
                face = cv2.equalizeHist(face)
                
                # Add to known faces
                known_faces.append(face)
                known_names.append(os.path.splitext(filename)[0])
                known_face_images.append(face)
    
    print(f"Loaded {len(known_names)} known faces")
    
    # Train the recognizer
    if len(known_faces) > 0:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Prepare labels (each person gets a unique numeric ID)
        labels = np.array([i for i in range(len(known_faces))])
        recognizer.train(known_face_images, labels)
        return known_faces, known_names, recognizer
    else:
        return known_faces, known_names, None

def detect_and_recognize_faces(frame, known_faces, known_names, recognizer):
    if recognizer is None or not known_faces:
        return frame
        
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_gray = gray[y:y+h, x:x+w]
        face_gray = cv2.resize(face_gray, (150, 150))
        face_gray = cv2.equalizeHist(face_gray)
        
        # Predict using the recognizer
        try:
            label, confidence = recognizer.predict(face_gray)
            
            # Lower confidence value means better match in LBPH
            if confidence < 70:  # Threshold for recognition
                name = known_names[label]
            else:
                name = "Unknown"
                
            # Add confidence percentage to display
            if name != "Unknown":
                confidence_percent = max(0, min(100, 100 - confidence))
                name = f"{name} ({confidence_percent:.1f}%)"
        except:
            name = "Unknown"
        
        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def main():
    # Load known faces and train recognizer
    known_faces, known_names, recognizer = load_known_faces()
    
    if not known_faces:
        print("No known faces found. Please add photos to the 'known_faces' directory.")
        print("Photos should be named with the person's name (e.g., 'john.jpg')")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera. Trying video file...")
        cap = cv2.VideoCapture("test.mp4")
        if not cap.isOpened():
            print("Error: Could not open camera or video file")
            return

    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame = detect_and_recognize_faces(frame, known_faces, known_names, recognizer)
        
        # Display result
        cv2.imshow('Face Recognition', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 