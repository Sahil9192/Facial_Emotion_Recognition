import os
from data_loader import get_data_generators
from model import build_emotion_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.join("data")
MODEL_SAVE_PATH = os.path.join("..", "saved_models", "emotion_model.h5")

# Hyperparameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 7

def main():
    print(" Loading data...")
    train_gen, test_gen = get_data_generators(BASE_DIR, IMG_SIZE, BATCH_SIZE)

    print(" Building model...")
    model = build_emotion_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)

    print(" Compiling model...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(" Training model...")
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    print(f" Model training complete. Best model saved to: {MODEL_SAVE_PATH}")

    history = model.history.history

# Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')  
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
