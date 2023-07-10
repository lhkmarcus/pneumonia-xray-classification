import os
import sys

from src.exceptions import CustomException
from src.logger import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tf messages
import tensorflow as tf

from dataclasses import dataclass


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


@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("images", "train")
    test_path: str = os.path.join("images", "test")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititate_data_ingestion(self):
        logging.info("Started data ingestion component.")
        try:
            train_ds, valid_ds = create_datasets(self.ingestion_config.train_path)
            test_ds = create_datasets(
                image_dir=self.ingestion_config.test_path, split=None, subset=None)
            logging.info("Loaded training and validation data as TensorFlow Datasets.")

            AUTOTUNE = tf.data.AUTOTUNE

            train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
            valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
            test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
            logging.info("Buffered prefetching configured for datasets.")

            logging.info("Exited data ingestion component.")
            return (train_ds, valid_ds, test_ds)

        except Exception as e: 
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().inititate_data_ingestion()