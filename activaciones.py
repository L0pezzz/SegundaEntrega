"""
activaciones.py
---------------
Funciones de activación y sus derivadas.

Funciones disponibles:
    relu(z)         -> max(0, z)
    relu_deriv(z)   -> 1 si z > 0, 0 si z <= 0
    sigmoid(z)      -> 1 / (1 + e^-z)
    sigmoid_deriv(z)-> σ(z) · (1 - σ(z))
    softmax(z)      -> e^zi / Σe^zj  (estabilizado numéricamente)
"""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit.
    Rango de salida: [0, +∞)
    Uso: capas ocultas.
    """
    return np.maximum(0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    """
    Derivada de ReLU respecto a z.
    Devuelve 1 donde z > 0, 0 en otro caso.
    """
    return (z > 0).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Función sigmoide.
    Rango de salida: (0, 1)
    Uso: capa de salida en clasificación binaria.

    Se aplica clip para estabilidad numérica (evitar overflow en exp).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    """
    Derivada de sigmoide: σ(z) · (1 - σ(z)).
    Valor máximo = 0.25 (en z = 0).
    """
    s = sigmoid(z)
    return s * (1.0 - s)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Función Softmax.
    Rango de salida: (0, 1) por componente, con Σ = 1 por fila.
    Uso: capa de salida en clasificación multiclase.

    Estabilización: se resta el máximo de cada fila antes de exp
    para evitar overflow, sin cambiar el resultado matemático.
    """
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)
