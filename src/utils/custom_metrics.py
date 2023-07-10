import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import recall_score, precision_score
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
        self._recall = tf.keras.metrics.Mean("macro_recall")
        self._fmeasure = tf.keras.metrics.Mean("macro_fmeasure")
        self._precision = tf.keras.metrics.Mean("macro_precision")

        self.epoch = 0
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, model.name))

    def on_epoch_end(self, batch, logs=None):
        self.epoch += 1
        predictions = self.model.predict(self.y_val)

        self._mcc.reset_state()
        self._aucroc.reset_state()
        self._recall.reset_state()
        self._fmeasure.reset_state()
        self._precision.reset_state()

        mcc = matthews_corrcoef(
            self.y_val, np.argmax(predictions, axis=-1))
        fmeasure = f1_score(
            self.y_val, np.argmax(predictions, axis=-1), average=None)
        recall = recall_score(
            self.y_val, np.argmax(predictions, axis=-1), average=None)
        precision = precision_score(
            self.y_val, np.argmax(predictions, axis=-1), average=None)
        aucroc = roc_auc_score(
            self.y_val, np.argmax(predictions, axis=-1), average=None)

        self._mcc.update_state(mcc)
        self._aucroc.update_state(aucroc)
        self._recall.update_state(recall)
        self._fmeasure.update_state(fmeasure)
        self._precision.update_state(precision)

        self._write_metric(
            self._mcc.name,
            self._mcc.result().numpy().astype(float))
        self._write_metric(
            self._aucroc.name,
            self._aucroc.result().numpy().astype(float))
        self._write_metric(
            self._recall.name,
            self._recall.result().numpy().astype(float))
        self._write_metric(
            self._fmeasure.name,
            self._fmeasure.result().numpy().astype(float))
        self._write_metric(
            self._precision.name,
            self._precision.result().numpy().astype(float))

    def _write_metric(self, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=self.epoch)
            self.summary_writer.flush()