import cv2
import numpy as np
import sys
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load model path (default: best performing one)
model_path = sys.argv[1] if len(sys.argv) > 1 else "../saved_models/emotion_model.h5"
print(f" Loading model from: {model_path}")
model = load_model(model_path)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']



# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)
print(" Webcam started... Press 'q' to quit.")

# Optional: For FPS calculation
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        cv2.imshow("ROI - what model sees", roi_resized)

        roi_normalized = roi_resized.astype("float32") / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=0)
        roi_reshaped = np.expand_dims(roi_reshaped, axis=-1)

    # Predict emotion
        prediction = model.predict(roi_reshaped, verbose=0)
        emotion_index = int(np.argmax(prediction))
        emotion = emotion_labels[emotion_index]

    # Draw box & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)


    # Optional: Display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the result
    cv2.imshow("Facial Emotion Recognition", frame)
    


    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
