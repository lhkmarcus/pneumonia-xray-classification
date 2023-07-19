import os
import wandb
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from sklearn.metrics import f1_score, matthews_corrcoef


class CustomLogCallbackWB(tf.keras.callbacks.Callback):
    def __init__(self, model, x_val, y_val):
        super().__init__()
        self.model = model
        self.x_val = x_val
        self.y_val = y_val

        # Instantiate standalone metrics:
        self._mcc = tf.keras.metrics.Mean(name="mcc")
        self._aucroc = tf.keras.metrics.AUC(name="auc")
        self._f1score = tf.keras.metrics.Mean(name="f1_score")

        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

        self._mcc.reset_state()
        self._aucroc.reset_state()
        self._f1score.reset_state()

        print("\nMaking predictions for Epoch {} ".format(self.epoch))
        predictions = self.model.predict(self.x_val)

        f1score = f1_score(self.y_val, np.argmax(predictions, axis=-1),
                            average=None)
        mcc = matthews_corrcoef(self.y_val, np.argmax(predictions, axis=-1))

        self._mcc.update_state(mcc)
        self._f1score.update_state(f1score)
        self._aucroc.update_state(self.y_val, np.argmax(predictions, axis=-1))

        print("Training loss   : {} ".format(
            self.epoch, logs["loss"]))
        print("Training acc    : {} ".format(
            self.epoch, logs["accuracy"]))
        print("Model MCC score : {} ".format(
            self.epoch, self._mcc.result().numpy()))
        print("Model AUC score : {} ".format(
            self.epoch, self._aucroc.result().numpy()))
        print("Model F1 score  : {} \n".format(
            self.epoch, self._f1score.result().numpy()))

        # Log metrics to WandB:
        wandb.log({"mcc": self._mcc.result().numpy(),
                   "aucroc": self._aucroc.result().numpy(),
                   "fmeasure": self._f1score.result().numpy(),
                   "loss": logs["loss"]})