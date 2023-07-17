import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tf messages
import tensorflow as tf


def create_datasets(image_dir=None, split=0.2, subset="both"):
    CLS_NAMES = ["normal", "bacteria", "virus"]
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    return tf.keras.utils.image_dataset_from_directory(
        directory=image_dir, 
        label_mode="int",
        class_names=CLS_NAMES,
        color_mode="rgb", 
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE, 
        shuffle=True,
        seed=42,
        validation_split=split, 
        subset=subset)