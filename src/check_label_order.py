from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set your dataset path here
data_dir = os.path.join("data", "Facial_Emotion_Data", "train")

# Create the generator
datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=1,
    shuffle=False
)

# Print the actual label to index mapping
print(" CLASS INDICES (Used during training):")
print(gen.class_indices)
