import os
import sys

from src.exceptions import CustomException
from src.logger import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tf messages
import tensorflow as tf

from dataclasses import dataclass

from src.utils import create_datasets

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