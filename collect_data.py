import cv2
import os
import time

DATASET_DIR = 'datasets'
REAL_DIR = os.path.join(DATASET_DIR, 'real')
SPOOF_DIR = os.path.join(DATASET_DIR, 'spoof')
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SPOOF_DIR, exist_ok=True)

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'r' to capture REAL face, 's' for SPOOF, 'q' to quit.")

def save_face(face_img, label_dir):
    ts = int(time.time() * 1000)
    path = os.path.join(label_dir, f"{ts}.jpg")
    cv2.imwrite(path, face_img)
    print(f"Saved: {path}")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('Data Collection', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and len(faces) > 0:
        for (x, y, w, h) in faces:
            save_face(frame[y:y+h, x:x+w], REAL_DIR)
    elif key == ord('s') and len(faces) > 0:
        for (x, y, w, h) in faces:
            save_face(frame[y:y+h, x:x+w], SPOOF_DIR)
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
