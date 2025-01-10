import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed # Considering moving to torch as Pyod was written in, to avoid two separate dependencies

from tsod.detectors import Detector
from tsod.features import create_dataset


def build_model(X_train, dropout_fraction=0.2, size=128):
    timesteps = X_train.shape[1]
    num_features = X_train.shape[2]
    print(timesteps, num_features)
    model = Sequential(
        [
            LSTM(size, input_shape=(timesteps, num_features)),
            Dropout(dropout_fraction),
            RepeatVector(timesteps),
            LSTM(size, return_sequences=True),
            Dropout(dropout_fraction),
            TimeDistributed(Dense(num_features)),
        ]
    )

    model.compile(loss="mae", optimizer="adam")
    return model


def fit(model, X_train, y_train=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min"
    )
    print(X_train.shape, y_train.shape)
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        shuffle=False,
    )
    return history


def calculate_loss(X, X_pred):
    min_vals = np.min(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-6)
    X_pred_norm = (X_pred - min_vals) / (max_vals - min_vals + 1e-6) 
    deviation = np.abs(X_pred_norm - X_norm) / (X_norm + 1e-6) # Now all values are between 0 and 1 
    loss = np.mean(deviation, axis=1)                           # And loss is calculated as the mean of the deviation
    return loss                                                # from previous itteration we avg over all features to get a single value for loss
                                                                # Maby focusing on only one time step would be better, as we are looking for anomalies in the data
                                                                
                                                


def detect(model, X, threshold=0.65):
    X_pred = model.predict(X)
    is_anomaly = calculate_loss(X, X_pred) > threshold  # I feel this should be scaled to the data
    return is_anomaly                                   
                                                                      

class AutoEncoderLSTM(Detector):
    def __init__(self, time_steps=3, threshold=0.65, size=128, dropout_fraction=0.2):
        # A 'second channel' could be added to support rain features or similar. maby allow the input to have more than one feature
        super().__init__()
        self._model = None
        self._history = None
        self._threshold = threshold
        self._dropout_fraction = dropout_fraction
        self._size = size
        self._time_steps = time_steps

    def _fit(self, data):
        X, y = self._create_features(data)
        self._model = build_model(X)  # TODO add scaler
        self._history = fit(self._model, X, y)
        return self

    def _detect(self, data, threshold=None):
        if threshold is not None: # added this here so we can play around with the threshold without having to retrain
            self._threshold = threshold 
        X, _ = self._create_features(data)
        is_anomaly = detect(self._model, X, self._threshold)
        return is_anomaly

    def _create_features(self, data):
        df = data.to_frame("timeseries")
        X, y = df[["timeseries"]], df.timeseries
        X, y = create_dataset(X, y, time_steps=self._time_steps)
        return X, y

    def __str__(self):
        return f"{self.__class__.__name__}({self._model})"
