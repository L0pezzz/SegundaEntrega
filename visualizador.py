"""
visualizador.py
---------------
Clase para graficar resultados del entrenamiento.

Clase disponible:
    Visualizador  ->  curvas de pérdida, accuracy y scatter 3D
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Nota: NO se usa backend 'Agg' para permitir ventanas interactivas en PC

COLORES_BINARIO    = ['#4F7FE0', '#E0604F']
COLORES_MULTICLASE = ['#4F7FE0', '#E0904F', '#6BCB77']
MARCADORES         = ['o', '^', 's']


class Visualizador:
    """
    Genera figuras de tres paneles para evaluar el entrenamiento:
      - Panel izquierdo  : curva de pérdida vs épocas
      - Panel central    : curva de accuracy vs épocas
      - Panel derecho    : scatter 3D con predicciones

    Parámetros:
        modo : 'binario' o 'multiclase'
    """

    def __init__(self, modo: str = 'binario'):
        if modo not in ('binario', 'multiclase'):
            raise ValueError("modo debe ser 'binario' o 'multiclase'")
        self.modo    = modo
        self.colores = COLORES_BINARIO if modo == 'binario' else COLORES_MULTICLASE

    def _graficar_perdida(self, ax, historial):
        epocas = range(1, len(historial) + 1)
        ax.plot(epocas, historial, color='#E0604F', linewidth=2)
        ax.set_xlabel('Época', fontsize=10)
        ax.set_ylabel('Pérdida (Cross-Entropy)', fontsize=10)
        ax.set_title('Curva de pérdida', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        ax.annotate(
            f"Final: {historial[-1]:.4f}",
            xy=(len(historial), historial[-1]),
            xytext=(-60, 15), textcoords='offset points',
            fontsize=9, color='#E0604F',
            arrowprops=dict(arrowstyle='->', color='#E0604F', lw=1),
        )

    def _graficar_accuracy(self, ax, historial):
        epocas  = range(1, len(historial) + 1)
        acc_pct = [a * 100 for a in historial]
        ax.plot(epocas, acc_pct, color='#4F7FE0', linewidth=2)
        ax.set_xlabel('Época', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Accuracy durante entrenamiento', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        ax.annotate(
            f"Final: {acc_pct[-1]:.1f}%",
            xy=(len(historial), acc_pct[-1]),
            xytext=(-60, -20), textcoords='offset points',
            fontsize=9, color='#4F7FE0',
            arrowprops=dict(arrowstyle='->', color='#4F7FE0', lw=1),
        )

    def _graficar_3d(self, ax, X, y_real, y_pred):
        for k in np.unique(y_real):
            color    = self.colores[int(k)]
            marcador = MARCADORES[int(k) % len(MARCADORES)]
            idx_ok   = (y_real == k) & (y_pred == k)
            idx_err  = (y_real == k) & (y_pred != k)
            if idx_ok.sum() > 0:
                ax.scatter(X[idx_ok, 0], X[idx_ok, 1], X[idx_ok, 2],
                           c=color, marker=marcador, s=30, alpha=0.7,
                           label=f'Clase {int(k)}')
            if idx_err.sum() > 0:
                ax.scatter(X[idx_err, 0], X[idx_err, 1], X[idx_err, 2],
                           c=color, marker='x', s=70, alpha=0.95,
                           linewidths=1.8, label=f'Clase {int(k)} (error)')
        ax.set_xlabel('x₁', fontsize=9)
        ax.set_ylabel('x₂', fontsize=9)
        ax.set_zlabel('x₃', fontsize=9)
        ax.set_title('Clasificación en 3D\n(× = error de predicción)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')

    def graficar(
        self,
        X,
        y_real_int,
        y_pred_int,
        historial_perdida,
        historial_accuracy,
        titulo,
        ruta_salida=None,
    ):
        """
        Genera la figura completa de 3 paneles.

        Parámetros:
            ruta_salida : (opcional) si se indica, guarda el PNG en esa ruta.
                          Si es None (por defecto), muestra la figura en pantalla.
        """
        fig = plt.figure(figsize=(17, 5))
        fig.suptitle(titulo, fontsize=13, fontweight='bold', y=1.01)

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        self._graficar_perdida(ax1, historial_perdida)
        self._graficar_accuracy(ax2, historial_accuracy)
        self._graficar_3d(ax3, X, y_real_int, y_pred_int)

        plt.tight_layout()

        if ruta_salida:
            plt.savefig(ruta_salida, dpi=130, bbox_inches='tight')
            plt.close()
            print(f"  Figura guardada en: {ruta_salida}")
        else:
            plt.show()   # abre ventana interactiva en tu PC