"""
datos.py
--------
Generación de datos sintéticos en espacio 3D para clasificación.

Clases disponibles:
    GeneradorBinario     -> 2 clases, nubes gaussianas separadas
    GeneradorMulticlase  -> 3 clases, nubes gaussianas separadas
"""

import numpy as np


class GeneradorBinario:
    """
    Genera puntos 3D de dos clases gaussianas separadas.

    Clase 0: centrada en (-1.5, -1.5, -1.5)
    Clase 1: centrada en ( 1.5,  1.5,  1.5)

    Parámetros:
        n         : número total de muestras
        ruido     : desviación estándar de cada nube
        semilla   : semilla aleatoria para reproducibilidad
    """

    def __init__(self, n: int = 400, ruido: float = 0.8, semilla: int = 42):
        self.n = n
        self.ruido = ruido
        self.semilla = semilla

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None  # shape (N, 1), valores {0, 1}

    def generar(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Genera y retorna (X, y).

        X : shape (N, 3)
        y : shape (N, 1), valores en {0.0, 1.0}
        """
        np.random.seed(self.semilla)
        n2 = self.n // 2

        X0 = np.random.randn(n2, 3) * self.ruido + np.array([-1.5, -1.5, -1.5])
        X1 = np.random.randn(n2, 3) * self.ruido + np.array([ 1.5,  1.5,  1.5])

        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(n2), np.ones(n2)]).reshape(-1, 1)

        # Mezclar aleatoriamente
        idx = np.random.permutation(self.n)
        self.X, self.y = X[idx], y[idx]
        return self.X, self.y

    def normalizar(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna los datos normalizados (media=0, std=1) por columna.
        Llama a generar() si aún no se ha generado.
        """
        if self.X is None:
            self.generar()
        self._media = self.X.mean(axis=0)
        self._std   = self.X.std(axis=0)
        X_norm = (self.X - self._media) / self._std
        return X_norm, self.y

    def resumen(self) -> None:
        if self.X is None:
            print("Datos no generados aún. Llama a generar() primero.")
            return
        print(f"GeneradorBinario | N={self.n} | ruido={self.ruido}")
        print(f"  Clase 0: {int((self.y == 0).sum())} muestras")
        print(f"  Clase 1: {int((self.y == 1).sum())} muestras")
        print(f"  X shape: {self.X.shape}")


class GeneradorMulticlase:
    """
    Genera puntos 3D de tres clases gaussianas separadas.

    Clase 0: centrada en ( 2.0,  0.0,  0.0)
    Clase 1: centrada en (-1.0,  1.7,  0.0)
    Clase 2: centrada en (-1.0, -1.7,  2.0)

    Parámetros:
        n         : número total de muestras (se redondea a múltiplo de 3)
        ruido     : desviación estándar de cada nube
        semilla   : semilla aleatoria para reproducibilidad
    """

    CENTROS = [
        np.array([ 2.0,  0.0,  0.0]),
        np.array([-1.0,  1.7,  0.0]),
        np.array([-1.0, -1.7,  2.0]),
    ]

    def __init__(self, n: int = 450, ruido: float = 0.7, semilla: int = 42):
        self.n = n
        self.ruido = ruido
        self.semilla = semilla

        self.X: np.ndarray | None = None
        self.y_onehot: np.ndarray | None = None  # shape (N, 3)
        self.y_int: np.ndarray | None = None      # shape (N,), valores {0,1,2}

    def generar(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera y retorna (X, y_onehot, y_int).

        X        : shape (N, 3)
        y_onehot : shape (N, 3)  — one-hot encoding
        y_int    : shape (N,)    — etiquetas enteras {0, 1, 2}
        """
        np.random.seed(self.semilla)
        n3 = self.n // 3

        bloques_X   = [np.random.randn(n3, 3) * self.ruido + c for c in self.CENTROS]
        bloques_int = [np.full(n3, k) for k in range(3)]

        X     = np.vstack(bloques_X)
        y_int = np.hstack(bloques_int)

        # One-hot encoding
        y_oh = np.zeros((len(y_int), 3))
        for i, k in enumerate(y_int):
            y_oh[i, int(k)] = 1.0

        # Mezclar
        idx = np.random.permutation(len(X))
        self.X        = X[idx]
        self.y_onehot = y_oh[idx]
        self.y_int    = y_int[idx]
        return self.X, self.y_onehot, self.y_int

    def normalizar(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna (X_norm, y_onehot, y_int) con X normalizado (media=0, std=1).
        """
        if self.X is None:
            self.generar()
        self._media = self.X.mean(axis=0)
        self._std   = self.X.std(axis=0)
        X_norm = (self.X - self._media) / self._std
        return X_norm, self.y_onehot, self.y_int

    def resumen(self) -> None:
        if self.X is None:
            print("Datos no generados aún. Llama a generar() primero.")
            return
        print(f"GeneradorMulticlase | N={self.n} | ruido={self.ruido}")
        for k in range(3):
            print(f"  Clase {k}: {int((self.y_int == k).sum())} muestras")
        print(f"  X shape: {self.X.shape}")
