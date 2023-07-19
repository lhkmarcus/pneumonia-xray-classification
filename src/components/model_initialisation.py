import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from src.logger import logging
from src.exceptions import CustomException


class PretrainedDenseNet:
    def __init__(self):
        super().__init__()
        self.img_size = (224, 224)
        self.img_shape = self.img_size + (3,)
        self.base_lr = 0.0001
        self.dropout = 0.2
        self.num_class = 3
        self.augment_layer = tf.keras.Sequential([
            tf.keras.layers.RandomContrast(factor=0.2, seed=42),
            tf.keras.layers.RandomBrightness(factor=0.2, seed=42),
            tf.keras.layers.RandomRotation(factor=0.1, seed=42)])

    def create_base_model(self):
        logging.info("[BEGIN] Base Model Initialisation")
        try:
            self.base_model = tf.keras.applications.densenet.DenseNet169(
                input_shape=self.img_shape, include_top=False, weights="imagenet")
            logging.info("Instantiated base model.")
            
            self.base_model.trainable = False
            logging.info("Froze base model trainable parameters.")
            logging.info("Base model initialisation successful.")
            logging.info("[OUT] Base Model Initialisation")
        
        except Exception as e:
            raise CustomException(e, sys)

    def build_model(self):
        logging.info("[ENTER] Network Build Component")
        try:
            inputs = tf.keras.Input(shape=self.img_shape)
            logging.info("Instantiated inputs.")
            
            x = self.augment_layer(inputs)
            logging.info("Instantiated data augmentation layer.")

            x = tf.keras.applications.densenet.preprocess_input(x)
            logging.info("Instantiated DenseNet preprocessing layer.")

            x = self.base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            logging.info("Instantiated global averaging layer.")
            
            x = tf.keras.layers.Dropout(self.dropout)(x)
            logging.info("Instantiated dropout layer.")

            outputs = tf.keras.layers.Dense(self.num_class, activation="softmax")(x)
            logging.info("Instantiated final classification (Dense) layer.")

            self.model = tf.keras.Model(inputs, outputs)
            logging.info("Pretrained model build successful.")
            logging.info("[EXIT] Network Build Component")
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def compile_model(self):
        logging.info("[BEGIN] Model Compilation")
        try:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"])
            logging.info("Model compilation successful.")

            logging.info("[EXIT] Model Compilation.")
            return self.model
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initialise_model(self):
        self.create_base_model()
        self.build_model()
        self.compile_model()
        logging.info("Model initialisation successful.")


if __name__ == "__main__":
    logging.info("Model initialisation test.")
    model = PretrainedDenseNet().initialise_model()