import os
import sys

from src.logger import logging
from src.exceptions import CustomException

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tf messages
import tensorflow as tf

from dataclasses import dataclass
from src.utils.create_datasets import create_datasets


@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifacts", "train")
    test_path: str = os.path.join("artifacts", "test")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititate_data_ingestion(self):
        logging.info("[BEGIN] Data Ingestion Component.")
        try:
            train_ds, valid_ds = create_datasets(self.ingestion_config.train_path)
            test_ds = create_datasets(
                image_dir=self.ingestion_config.test_path, split=None, subset=None)
            logging.info("Loaded training and validation data.")

            AUTOTUNE = tf.data.AUTOTUNE

            train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
            valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
            test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
            logging.info("Configured prefetch buffer for datasets.")

            logging.info("[EXIT] Data Ingestion Component.")
            
            logging.info("Data ingestion successful.")
            return (train_ds, valid_ds, test_ds)

        except Exception as e: 
            raise CustomException(e, sys)


# Test data ingestion component:
if __name__ == "__main__":
    logging.info("Data ingestion test.")
    DataIngestion().inititate_data_ingestion()