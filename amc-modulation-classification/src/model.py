"""1D CNN for Automatic Modulation Classification (Keras / TensorFlow)."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(n_classes=11, n_samples=128):
    """Input: I/Q burst shaped (128, 2) -> softmax over modulation classes."""
    inputs = keras.Input(shape=(n_samples, 2), name="iq")
    x = inputs
    for filters in (64, 64, 128, 128):
        x = layers.Conv1D(filters, 7, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="modulation")(x)

    model = keras.Model(inputs, outputs, name="amc_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
