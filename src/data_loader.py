import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir, img_size=(48, 48), batch_size=64):
    train_dir = os.path.join(base_dir, "Facial_Emotion_Data", "train")
    test_dir = os.path.join(base_dir, "Facial_Emotion_Data", "test")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    # Only rescaling for validation/test
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator
