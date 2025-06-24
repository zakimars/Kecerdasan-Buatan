import cv2
import numpy as np
import os
from keras.models import load_model
from datetime import datetime
import time

model = load_model('emotion_model.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (255, 0, 255),
    'Happy': (0, 255, 255),
    'Sad': (255, 255, 0),
    'Surprise': (255, 0, 0),
    'Neutral': (200, 200, 200)
}

os.makedirs("snapshots", exist_ok=True)

def preprocess_face(gray, x, y, w, h):
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi = roi_gray.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)
    return roi

def detect_emotion(roi):
    prediction = model.predict(roi, verbose=0)[0]
    return emotion_labels[np.argmax(prediction)]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = preprocess_face(gray, x, y, w, h)
        label = detect_emotion(roi)
        color = emotion_colors[label]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Deteksi Emosi", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and len(faces) > 0:
        filename = f"snapshots/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Snapshot saved to {filename}")

cap.release()
cv2.destroyAllWindows()
