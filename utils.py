import cv2
import numpy as np

def preprocess_frame_for_model(frame, target_size=(128,128)):
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, 0)
    return img
