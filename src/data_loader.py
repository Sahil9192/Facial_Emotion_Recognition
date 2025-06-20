import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir, img_size=(48, 48), batch_size=64):
    train_dir = os.path.join(base_dir, "Facial_Emotion_Data", "train")
    test_dir = os.path.join(base_dir, "Facial_Emotion_Data", "test")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'Facial_Emotion_Data', 'train'),
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'Facial_Emotion_Data', 'test'),
        target_size=img_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, test_generator
