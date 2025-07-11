import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_data_generators
from model import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))             # /src
BASE_DIR = os.path.join(PROJECT_ROOT, "..", "data")                   # ../data
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "saved_models")
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "emotion_model.h5")
FINAL_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "final_emotion_model.h5")


IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 7

def main():
    print(" Loading data...")
    train_gen, test_gen = get_data_generators(BASE_DIR, IMG_SIZE, BATCH_SIZE)
    print("Class label order:", train_gen.class_indices)


    print(" Building model...")
    model = build_model()

    print(" Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(" Calculating class weights...")
    y_train = train_gen.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    print(" Setting up callbacks...")
    checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        verbose=1,
        min_lr=1e-6
    )

    callbacks = [checkpoint, early_stop, lr_reduce]

    print(" Starting training...")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )

    print(f" Model training complete. Best model saved to: {BEST_MODEL_PATH}")

    
    model.save(FINAL_MODEL_PATH)
    print(f" Final model also saved to: {FINAL_MODEL_PATH}")

    
    plot_training_curves(history)


def plot_training_curves(history):
    history_dict = history.history

    
    plt.figure(figsize=(10, 5))
    plt.plot(history_dict['accuracy'], label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    plt.figure(figsize=(10, 5))
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
