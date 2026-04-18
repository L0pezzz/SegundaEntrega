"""
perdida.py
----------
Funciones de costo (loss functions).

Funciones disponibles:
    entropia_cruzada_binaria(y, y_pred)     -> BCE loss  (escalar)
    entropia_cruzada_categorica(y, y_pred)  -> CCE loss  (escalar)
"""

import numpy as np

# Pequeña constante para evitar log(0)
_EPS = 1e-15


def entropia_cruzada_binaria(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary Cross-Entropy (BCE).

    Fórmula:
        L = -1/N · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]

    Parámetros:
        y      : etiquetas reales, shape (N, 1), valores en {0, 1}
        y_pred : probabilidades predichas, shape (N, 1), valores en (0, 1)

    Retorna:
        Pérdida promedio (escalar).
    """
    loss = -(
        y * np.log(y_pred + _EPS) +
        (1.0 - y) * np.log(1.0 - y_pred + _EPS)
    )
    return float(np.mean(loss))


def entropia_cruzada_categorica(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Categorical Cross-Entropy (CCE).

    Fórmula:
        L = -1/N · Σ_i Σ_k [ y_ik · log(ŷ_ik) ]

    Como y es one-hot, esto equivale a:
        L = -1/N · Σ_i log(ŷ_{i, clase_correcta})

    Parámetros:
        y      : etiquetas one-hot, shape (N, K)
        y_pred : probabilidades softmax, shape (N, K)

    Retorna:
        Pérdida promedio (escalar).
    """
    N = len(y)
    loss = -np.sum(y * np.log(y_pred + _EPS)) / N
    return float(loss)
