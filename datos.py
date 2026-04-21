import numpy as np

DISTRIBUCIONES_VALIDAS = ('normal', 'exponencial', 'laplace', 'uniforme')


def _ruido(n: int, dims: int, distribucion: str, escala: float) -> np.ndarray:
    """
    Genera n × dims muestras de ruido centrado en 0 con la dispersión dada.

    Se centra la exponencial restando su media (escala) para que todas
    las distribuciones tengan media 0 y dispersión comparable.
    Para uniforme, el rango ±escala·√3 iguala la desviación estándar
    a la del resto de distribuciones.

    Args:
        n           : número de muestras
        dims        : dimensiones (3 en nuestro caso)
        distribucion: 'normal' | 'exponencial' | 'laplace' | 'uniforme'
        escala      : controla la dispersión (equivalente a sigma en normal)

    Returns:
        Array de forma (n, dims)
    """
    if distribucion not in DISTRIBUCIONES_VALIDAS:
        raise ValueError(
            f"Distribución '{distribucion}' no válida. "
            f"Usa una de: {DISTRIBUCIONES_VALIDAS}"
        )

    if distribucion == 'normal':
        # N(0, escala)
        return np.random.randn(n, dims) * escala

    elif distribucion == 'exponencial':
        # Exp(escala) centrada en 0: media = escala, así restamos escala
        return np.random.exponential(scale=escala, size=(n, dims)) - escala

    elif distribucion == 'laplace':
        # Laplace(0, escala/√2): desviación estándar = escala
        return np.random.laplace(loc=0, scale=escala / np.sqrt(2), size=(n, dims))

    elif distribucion == 'uniforme':
        # U(-a, a) donde a = escala·√3 → std = escala
        a = escala * np.sqrt(3)
        return np.random.uniform(-a, a, size=(n, dims))


class GeneradorBinario:
    """
    Genera puntos 3D de dos clases gaussianas (o con otra distribución).

    Clase 0: centrada en (-1.5, -1.5, -1.5)
    Clase 1: centrada en ( 1.5,  1.5,  1.5)

    Args:
        n           : número TOTAL de muestras
        ruido       : dispersión de cada nube (equivale a sigma en distribución normal)
        semilla     : semilla aleatoria
        distribucion: 'normal' | 'exponencial' | 'laplace' | 'uniforme'
    """

    CENTRO_0 = np.array([-1.5, -1.5, -1.5])
    CENTRO_1 = np.array([ 1.5,  1.5,  1.5])

    def __init__(
        self,
        n: int = 400,
        ruido: float = 0.8,
        semilla: int = 42,
        distribucion: str = 'normal',
    ):
        self.n            = n
        self.ruido        = ruido
        self.semilla      = semilla
        self.distribucion = distribucion

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None  # shape (N, 1)

    def generar(self) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.semilla)
        n2 = self.n // 2

        X0 = _ruido(n2, 3, self.distribucion, self.ruido) + self.CENTRO_0
        X1 = _ruido(n2, 3, self.distribucion, self.ruido) + self.CENTRO_1

        X  = np.vstack([X0, X1])
        y  = np.hstack([np.zeros(n2), np.ones(n2)]).reshape(-1, 1)

        idx = np.random.permutation(self.n)
        self.X, self.y = X[idx], y[idx]
        return self.X, self.y

    def normalizar(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            self.generar()
        self._media = self.X.mean(axis=0)
        self._std   = self.X.std(axis=0) + 1e-8
        return (self.X - self._media) / self._std, self.y

    def resumen(self) -> None:
        if self.X is None:
            print("Datos no generados aún.")
            return
        print(f"GeneradorBinario | N={self.n} | ruido={self.ruido} | dist={self.distribucion}")
        print(f"  Clase 0: {int((self.y == 0).sum())} muestras")
        print(f"  Clase 1: {int((self.y == 1).sum())} muestras")


class GeneradorMulticlase:
    """
    Genera puntos 3D de tres clases con distribución configurable.

    Clase 0: centrada en ( 2.0,  0.0,  0.0)
    Clase 1: centrada en (-1.0,  1.7,  0.0)
    Clase 2: centrada en (-1.0, -1.7,  2.0)

    Args:
        n           : número TOTAL de muestras (se redondea a múltiplo de 3)
        ruido       : dispersión de cada nube
        semilla     : semilla aleatoria
        distribucion: 'normal' | 'exponencial' | 'laplace' | 'uniforme'
    """

    CENTROS = [
        np.array([ 2.0,  0.0,  0.0]),
        np.array([-1.0,  1.7,  0.0]),
        np.array([-1.0, -1.7,  2.0]),
    ]

    def __init__(
        self,
        n: int = 450,
        ruido: float = 0.7,
        semilla: int = 42,
        distribucion: str = 'normal',
    ):
        self.n            = n
        self.ruido        = ruido
        self.semilla      = semilla
        self.distribucion = distribucion

        self.X:        np.ndarray | None = None
        self.y_onehot: np.ndarray | None = None
        self.y_int:    np.ndarray | None = None

    def generar(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.semilla)
        n3 = self.n // 3

        bloques_X   = [_ruido(n3, 3, self.distribucion, self.ruido) + c
                       for c in self.CENTROS]
        bloques_int = [np.full(n3, k) for k in range(3)]

        X     = np.vstack(bloques_X)
        y_int = np.hstack(bloques_int)

        y_oh = np.zeros((len(y_int), 3))
        for i, k in enumerate(y_int):
            y_oh[i, int(k)] = 1.0

        idx = np.random.permutation(len(X))
        self.X        = X[idx]
        self.y_onehot = y_oh[idx]
        self.y_int    = y_int[idx]
        return self.X, self.y_onehot, self.y_int

    def normalizar(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.X is None:
            self.generar()
        self._media = self.X.mean(axis=0)
        self._std   = self.X.std(axis=0) + 1e-8
        X_norm = (self.X - self._media) / self._std
        return X_norm, self.y_onehot, self.y_int

    def resumen(self) -> None:
        if self.X is None:
            print("Datos no generados aún.")
            return
        print(f"GeneradorMulticlase | N={self.n} | ruido={self.ruido} | dist={self.distribucion}")
        for k in range(3):
            print(f"  Clase {k}: {int((self.y_int == k).sum())} muestras")