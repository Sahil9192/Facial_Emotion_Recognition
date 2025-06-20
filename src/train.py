import os
from data_loader import get_data_generators
from model import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
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
    model = build_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)

    print(" Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(" Training model...")

    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    y_train = train_gen.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1)
    callbacks = [checkpoint, early_stop, lr_reduce]

    # Training
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    print(f" Model training complete. Best model saved to: {MODEL_SAVE_PATH}")

    # Visualization
    history_dict = history.history

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history_dict['accuracy'], label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
