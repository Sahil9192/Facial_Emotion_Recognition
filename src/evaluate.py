import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import get_data_generators

# Paths
BASE_DIR = os.path.join("data")
MODEL_PATH = os.path.join("..", "saved_models", "emotion_model.h5")
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def evaluate_model():
    # Load test data
    _, test_gen = get_data_generators(BASE_DIR, img_size=(48, 48), batch_size=32)

    # Load model
    model = load_model(MODEL_PATH)

    # Predict
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
