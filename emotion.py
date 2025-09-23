import cv2
from deepface import DeepFace
import os
import random

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Map emotions to folder paths (update with your own folders)
emotion_folders = {
    "happy": "images/happy",
    "sad": "images/sad",
    "angry": "images/angry",
    "surprise": "images/surprise",
    "fear": "images/fear",
    "neutral": "images/neutral",
    "disgust": "images/disgust"
}

last_emotion = None  # track last detected emotion

# Desired display size for images
display_size = (400, 400)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Show random image only when emotion changes
        if emotion != last_emotion and emotion in emotion_folders:
            folder = emotion_folders[emotion]
            img_files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            if img_files:
                img_path = os.path.join(folder, random.choice(img_files))
                img_to_show = cv2.imread(img_path)

                if img_to_show is not None:
                    # Resize image to fixed size
                    img_resized = cv2.resize(img_to_show, display_size, interpolation=cv2.INTER_AREA)
                    cv2.imshow("Emotion Image", img_resized)

            last_emotion = emotion

        # Draw rectangle + emotion label on live feed
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show live camera feed
    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
