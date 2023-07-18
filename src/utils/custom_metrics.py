import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef


# Create custom metrics
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, x_val, y_val, log_dir):
        super().__init__()
        self.model = model
        self.x_val = x_val
        self.y_val = y_val

        # Instantiate macro-averaged metrics
        self._mcc = tf.keras.metrics.Metric("mcc")
        self._aucroc = tf.keras.metrics.Mean("macro_aucroc")
        self._fmeasure = tf.keras.metrics.Mean("macro_fmeasure")

        self.epoch = 0

    def on_epoch_end(self, batch, logs=None):
        self.epoch += 1
        predictions = self.model.predict(self.y_val)

        self._mcc.reset_state()
        self._aucroc.reset_state()
        self._fmeasure.reset_state()

        mcc = matthews_corrcoef(
            self.y_val, np.argmax(predictions, axis=-1))
        fmeasure = f1_score(
            self.y_val, np.argmax(predictions, axis=-1), average=None)
        aucroc = roc_auc_score(
            self.y_val, np.argmax(predictions, axis=-1), average=None)

        self._mcc.update_state(mcc)
        self._aucroc.update_state(aucroc)
        self._fmeasure.update_state(fmeasure)

        # Append the values to the weights and biases log:
        