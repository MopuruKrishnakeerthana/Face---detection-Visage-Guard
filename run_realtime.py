import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = 'models/spoof_detector.h5'
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE = (128, 128)

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(HAAR_PATH)
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise SystemExit("Cannot open webcam")

REAL_LABEL = 'Real'
SPOOF_LABEL = 'Spoof'

def preprocess_face(face):
    face = cv2.resize(face, IMG_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    return np.expand_dims(face, 0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        pred = model.predict(preprocess_face(face_img), verbose=0)[0][0]

        # Correct mapping based on training class
        if pred < 0.5:
            label = REAL_LABEL
            color = (0, 255, 0)
            conf = (1 - pred) * 100
        else:
            label = SPOOF_LABEL
            color = (0, 0, 255)
            conf = pred * 100

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({conf:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Anti-Spoofing Live', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
