import cv2
import numpy as np

def detect_faces(frame):
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def main():
    # Try to open the default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera. Trying video file...")
        # If camera fails, try to use a video file
        cap = cv2.VideoCapture("test.mp4")
        if not cap.isOpened():
            print("Error: Could not open camera or video file")
            return

    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        frame = detect_faces(frame)
        
        # Display the result
        cv2.imshow('Face Detection', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 