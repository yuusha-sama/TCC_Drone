import numpy as np
try:
    import tensorflow as tf
except Exception as e:
    tf = None
    _ERR = e

class KerasBackend:
    """Backend para modelos Keras (.h5 / .keras)."""
    def __init__(self, model_path: str):
        if tf is None:
            raise RuntimeError(f"TensorFlow/Keras não disponível: {_ERR}")
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = tuple(self.model.inputs[0].shape)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[..., None]
        y = self.model.predict(x, verbose=0)
        return y
