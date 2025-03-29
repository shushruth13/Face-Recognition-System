import cv2
import os

# Create the folder if it doesn't exist
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 's' to save a face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and len(faces) == 1:  # Save if exactly 1 face is detected
        face_img = frame[y:y+h, x:x+w]  # Crop the face
        name = input("Enter the person's name: ").strip()
        cv2.imwrite(f"known_faces/{name}.jpg", face_img)
        print(f"Saved {name}.jpg!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()