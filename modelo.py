"""
modelo.py
---------
Clase base MLP (Perceptrón Multicapa).

Implementa forward pass, backward pass (backpropagation)
y entrenamiento con mini-batch SGD desde cero con NumPy.

Clase disponible:
    MLP   ->  red neuronal configurable por lista de capas
"""

import numpy as np
from activaciones import relu, relu_deriv, sigmoid, softmax
from perdida import entropia_cruzada_binaria, entropia_cruzada_categorica


class MLP:
    """
    Perceptrón Multicapa (MLP) para clasificación.

    Parámetros:
        capas      : lista de enteros que define el tamaño de cada capa.
                     Ej: [3, 4, 1] = entrada de 3, oculta de 4, salida de 1.
        modo       : 'binario' o 'multiclase'
        lr         : tasa de aprendizaje (learning rate)
        epocas     : número de épocas de entrenamiento
        batch_size : tamaño del mini-batch

    Atributos públicos tras entrenar:
        historial_perdida  : list[float]  — pérdida por época
        historial_accuracy : list[float]  — accuracy por época (0 a 1)
        W                  : list[ndarray] — pesos por capa
        b                  : list[ndarray] — sesgos por capa
    """

    def __init__(
        self,
        capas: list[int],
        modo: str = 'binario',
        lr: float = 0.05,
        epocas: int = 300,
        batch_size: int = 32,
    ):
        if modo not in ('binario', 'multiclase'):
            raise ValueError("modo debe ser 'binario' o 'multiclase'")

        self.capas      = capas
        self.modo       = modo
        self.lr         = lr
        self.epocas     = epocas
        self.batch_size = batch_size

        self.historial_perdida:  list[float] = []
        self.historial_accuracy: list[float] = []

        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self._inicializar_pesos()

    # ─────────────────────────────────────────────────────
    # INICIALIZACIÓN DE PESOS
    # ─────────────────────────────────────────────────────

    def _inicializar_pesos(self) -> None:
        """
        Inicialización He para capas con ReLU.
            W ~ N(0, sqrt(2 / n_entrada))

        Mantiene la varianza de las activaciones constante entre capas,
        previniendo el desvanecimiento del gradiente.
        """
        for l in range(len(self.capas) - 1):
            n_in  = self.capas[l]
            n_out = self.capas[l + 1]
            escala = np.sqrt(2.0 / n_in)
            self.W.append(np.random.randn(n_in, n_out) * escala)
            self.b.append(np.zeros((1, n_out)))

    # ─────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante por todas las capas.

        Para cada capa l:
            z[l] = A[l-1] · W[l] + b[l]    (pre-activación)
            A[l] = f(z[l])                  (activación)

        Capas ocultas : ReLU
        Capa de salida: sigmoide (binario) o softmax (multiclase)

        Almacena self.A y self.Z para usar en backpropagation.

        Retorna:
            A[-1] — predicciones, shape (N, n_salida)
        """
        self.A = [X]
        self.Z = []

        n_capas = len(self.W)
        for l in range(n_capas):
            Z_l = self.A[l] @ self.W[l] + self.b[l]
            self.Z.append(Z_l)

            es_ultima = (l == n_capas - 1)
            if not es_ultima:
                self.A.append(relu(Z_l))
            elif self.modo == 'binario':
                self.A.append(sigmoid(Z_l))
            else:
                self.A.append(softmax(Z_l))

        return self.A[-1]

    # ─────────────────────────────────────────────────────
    # BACKWARD PASS
    # ─────────────────────────────────────────────────────

    def _backward(
        self, y: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Retropropagación usando la regla de la cadena.

        Gradiente de la capa de salida
        (entropía cruzada + sigmoide/softmax se simplifica a):
            δ[L] = ŷ - y

        Gradiente de capas ocultas:
            δ[l] = (δ[l+1] · W[l+1]ᵀ) ⊙ ReLU'(z[l])

        Gradientes de pesos y sesgos:
            ∂L/∂W[l] = A[l-1]ᵀ · δ[l] / N
            ∂L/∂b[l] = media(δ[l], eje=0)

        Retorna:
            (dW, db) — listas de gradientes por capa
        """
        N  = len(y)
        L  = len(self.W)
        dW = [None] * L
        db = [None] * L

        # Delta en la capa de salida
        delta = self.A[-1] - y

        for l in reversed(range(L)):
            dW[l] = (self.A[l].T @ delta) / N
            db[l] = delta.mean(axis=0, keepdims=True)

            # Propagar delta hacia la capa anterior (si no es la primera)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_deriv(self.Z[l - 1])

        return dW, db

    # ─────────────────────────────────────────────────────
    # ACTUALIZACIÓN DE PESOS (SGD)
    # ─────────────────────────────────────────────────────

    def _actualizar_pesos(
        self,
        dW: list[np.ndarray],
        db: list[np.ndarray],
    ) -> None:
        """
        Gradiente descendente estocástico (SGD).
            W[l] ← W[l] - lr · ∂L/∂W[l]
            b[l] ← b[l] - lr · ∂L/∂b[l]
        """
        for l in range(len(self.W)):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]

    # ─────────────────────────────────────────────────────
    # MÉTRICAS
    # ─────────────────────────────────────────────────────

    def _calcular_perdida(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        if self.modo == 'binario':
            return entropia_cruzada_binaria(y, y_pred)
        return entropia_cruzada_categorica(y, y_pred)

    def _calcular_accuracy(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        if self.modo == 'binario':
            clases_pred = (y_pred >= 0.5).astype(int)
            return float(np.mean(clases_pred == y))
        return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)))

    # ─────────────────────────────────────────────────────
    # ENTRENAMIENTO
    # ─────────────────────────────────────────────────────

    def entrenar(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Ciclo de entrenamiento completo con mini-batches.

        En cada época:
          1. Mezcla aleatoria de los datos
          2. Divide en mini-batches
          3. Forward → Backward → Actualizar pesos
          4. Registra pérdida y accuracy sobre el conjunto completo

        Parámetros:
            X       : datos de entrada, shape (N, 3)
            y       : etiquetas, shape (N, 1) para binario o (N, K) para multiclase
            verbose : si True, imprime métricas cada 50 épocas
        """
        N = len(X)
        for epoca in range(self.epocas):
            # Mezclar en cada época para evitar sesgos de orden
            idx = np.random.permutation(N)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            # Mini-batches
            for start in range(0, N, self.batch_size):
                end   = start + self.batch_size
                X_b   = X_shuffled[start:end]
                y_b   = y_shuffled[start:end]
                self._forward(X_b)
                dW, db = self._backward(y_b)
                self._actualizar_pesos(dW, db)

            # Registrar métricas sobre todos los datos
            y_pred = self._forward(X)
            perdida  = self._calcular_perdida(y, y_pred)
            accuracy = self._calcular_accuracy(y, y_pred)
            self.historial_perdida.append(perdida)
            self.historial_accuracy.append(accuracy)

            if verbose and (epoca + 1) % 50 == 0:
                print(
                    f"  Época {epoca + 1:4d}/{self.epocas} | "
                    f"Pérdida: {perdida:.4f} | "
                    f"Accuracy: {accuracy * 100:.1f}%"
                )

    # ─────────────────────────────────────────────────────
    # PREDICCIÓN
    # ─────────────────────────────────────────────────────

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna la clase predicha para cada muestra.

        Binario     : 0 o 1  — shape (N,)
        Multiclase  : 0, 1 o 2 — shape (N,)
        """
        y_pred = self._forward(X)
        if self.modo == 'binario':
            return (y_pred >= 0.5).astype(int).flatten()
        return np.argmax(y_pred, axis=1)

    def predecir_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna las probabilidades crudas de cada clase.

        Binario    : shape (N, 1)
        Multiclase : shape (N, K)
        """
        return self._forward(X)

    # ─────────────────────────────────────────────────────
    # RESUMEN
    # ─────────────────────────────────────────────────────

    def resumen(self) -> None:
        """Imprime la arquitectura y el número de parámetros por capa."""
        print(f"\nArquitectura MLP — modo: {self.modo}")
        print(f"{'Capa':<10} {'Entrada':<10} {'Salida':<10} {'Parámetros':<12}")
        print("-" * 44)
        total = 0
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            params = W.size + b.size
            total += params
            nombre = "Oculta" if l < len(self.W) - 1 else "Salida"
            print(f"{nombre} {l+1:<4} {W.shape[0]:<10} {W.shape[1]:<10} {params:<12}")
        print("-" * 44)
        print(f"{'Total':<32} {total:<12}")
