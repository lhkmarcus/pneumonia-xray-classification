import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from src.logger import logging
from src.exceptions import CustomException


class PretrainedDenseNet:
    def __init__(self):
        super().__init__()
        self.IMG_SIZE = (224, 224)
        self.IMG_SHAPE = self.IMG_SIZE + (3,)
        self.BASE_LR = 0.0001

    def create_base_model(self):
        logging.info("Started model initialisation component.")
        try:
            self.base_model = tf.keras.applications.densenet.DenseNet169(
                input_shape=self.IMG_SHAPE,
                include_top=False,
                weights="imagenet")
            logging.info("Instantiated base model.")
            
            self.base_model.trainable = False
            logging.info("Froze base model trainable parameters.")
        
        except Exception as e:
            raise CustomException(e, sys)

    def build_model(self):
        try:
            preprocess_input = tf.keras.applications.densenet.preprocess_input

            global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
            clf_layer = tf.keras.layers.Dense(3, activation="softmax")

            inputs = tf.keras.Input(shape=self.IMG_SHAPE)

            x = preprocess_input(inputs)
            logging.info("Instantiated DenseNet preprocessing layer.")

            x = self.base_model(x, training=False)
            x = global_avg_layer(x)
            logging.info("Instantiated global averaging layer.")
            
            x = tf.keras.layers.Dropout(0.2)(x)
            logging.info("Instantiated dropout layer with 0.2 rate.")

            outputs = clf_layer(x)
            logging.info("Instantiated final three-node dense layer with softmax activation.")

            self.model = tf.keras.Model(inputs, outputs)
            logging.info("Finished pretrained model build.")
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def compile_model(self):
        try:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.BASE_LR),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"])
            logging.info("Compiled model.")

            logging.info("Exited pretrained model initialisation component.")
            return self.model
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initialise_model(self):
        self.create_base_model()
        self.build_model()
        self.compile_model()


if __name__ == "__main__":
    model = PretrainedDenseNet().initialise_model()