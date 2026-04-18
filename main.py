"""
main.py
-------
Punto de entrada del proyecto.

Orquesta ambos escenarios de clasificación en 3D:
  - Escenario 1: Binario      [3 → 4 → 1]
  - Escenario 2: Multiclase   [3 → 8 → 3]

Uso:
    python main.py            -> corre ambos escenarios
    python main.py --s1       -> solo escenario 1
    python main.py --s2       -> solo escenario 2
"""

import argparse
import time
import numpy as np

from datos        import GeneradorBinario, GeneradorMulticlase
from modelo       import MLP
from visualizador import Visualizador


# ─────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────

SEMILLA    = 42
EPOCAS     = 300
LR         = 0.05
BATCH_SIZE = 32

CONFIG = {
    'escenario1': {
        'capas'      : [3, 4, 1],
        'modo'       : 'binario',
        'n_datos'    : 400,
        'ruido'      : 2,
        'ruta_salida': None,   # None = mostrar en pantalla | 'ruta/archivo.png' = guardar
        'titulo'     : 'Escenario 1 — Clasificación Binaria [3 → 4 → 1]',
    },
    'escenario2': {
        'capas'      : [3, 8, 3],
        'modo'       : 'multiclase',
        'n_datos'    : 450,
        'ruido'      : 0.7,
        'ruta_salida': None,   # None = mostrar en pantalla | 'ruta/archivo.png' = guardar
        'titulo'     : 'Escenario 2 — Clasificación Multiclase [3 → 8 → 3]',
    },
}


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────

def separador(titulo: str) -> None:
    ancho = 55
    print("\n" + "=" * ancho)
    print(f"  {titulo}")
    print("=" * ancho)


def matriz_confusion(y_real: np.ndarray, y_pred: np.ndarray, n_clases: int) -> None:
    """Imprime la matriz de confusión en consola."""
    print(f"\n  {'':10}", end="")
    for k in range(n_clases):
        print(f"{'Pred ' + str(k):>8}", end="")
    print()
    for k_real in range(n_clases):
        print(f"  Real {k_real}    ", end="")
        for k_pred in range(n_clases):
            cuenta = int(((y_real == k_real) & (y_pred == k_pred)).sum())
            print(f"{cuenta:>8}", end="")
        print()


def imprimir_resumen(nombre: str, accuracy: float, aciertos: int,
                     total: int, errores: int, tiempo: float) -> dict:
    """Imprime y retorna el resumen de un escenario."""
    print(f"\n  Accuracy  : {accuracy:.2f}%")
    print(f"  Aciertos  : {aciertos} / {total}")
    print(f"  Errores   : {errores}")
    print(f"  Tiempo    : {tiempo:.2f}s")
    return {
        'nombre'  : nombre,
        'accuracy': accuracy,
        'aciertos': aciertos,
        'errores' : errores,
        'tiempo'  : tiempo,
    }


# ─────────────────────────────────────────────────────
# ESCENARIO 1 — BINARIO
# ─────────────────────────────────────────────────────

def correr_escenario1() -> dict:
    cfg = CONFIG['escenario1']
    separador("ESCENARIO 1 — Clasificación Binaria  [3 → 4 → 1]")

    # 1. Datos
    gen = GeneradorBinario(n=cfg['n_datos'], ruido=cfg['ruido'], semilla=SEMILLA)
    gen.generar()
    gen.resumen()
    X, y = gen.normalizar()

    # 2. Modelo
    np.random.seed(SEMILLA)
    red = MLP(capas=cfg['capas'], modo=cfg['modo'],
              lr=LR, epocas=EPOCAS, batch_size=BATCH_SIZE)
    red.resumen()

    # 3. Entrenamiento
    print("\n  Entrenando...")
    t0 = time.time()
    red.entrenar(X, y, verbose=True)
    tiempo = time.time() - t0

    # 4. Evaluación
    y_pred = red.predecir(X)
    y_real = y.flatten().astype(int)
    aciertos = int((y_pred == y_real).sum())
    errores  = int((y_pred != y_real).sum())
    accuracy = aciertos / len(y_real) * 100

    print("\n  Matriz de confusión:")
    matriz_confusion(y_real, y_pred, n_clases=2)
    resumen = imprimir_resumen("Binario", accuracy, aciertos, len(y_real), errores, tiempo)

    # 5. Visualización
    Visualizador(modo='binario').graficar(
        X                  = X,
        y_real_int         = y_real,
        y_pred_int         = y_pred,
        historial_perdida  = red.historial_perdida,
        historial_accuracy = red.historial_accuracy,
        titulo             = cfg['titulo'],
        ruta_salida        = cfg['ruta_salida'],
    )

    return resumen


# ─────────────────────────────────────────────────────
# ESCENARIO 2 — MULTICLASE
# ─────────────────────────────────────────────────────

def correr_escenario2() -> dict:
    cfg = CONFIG['escenario2']
    separador("ESCENARIO 2 — Clasificación Multiclase  [3 → 8 → 3]")

    # 1. Datos
    gen = GeneradorMulticlase(n=cfg['n_datos'], ruido=cfg['ruido'], semilla=SEMILLA)
    gen.generar()
    gen.resumen()
    X, y_oh, y_int = gen.normalizar()

    # 2. Modelo
    np.random.seed(SEMILLA)
    red = MLP(capas=cfg['capas'], modo=cfg['modo'],
              lr=LR, epocas=EPOCAS, batch_size=BATCH_SIZE)
    red.resumen()

    # 3. Entrenamiento
    print("\n  Entrenando...")
    t0 = time.time()
    red.entrenar(X, y_oh, verbose=True)
    tiempo = time.time() - t0

    # 4. Evaluación
    y_pred = red.predecir(X)
    aciertos = int((y_pred == y_int).sum())
    errores  = int((y_pred != y_int).sum())
    accuracy = aciertos / len(y_int) * 100

    print("\n  Matriz de confusión:")
    matriz_confusion(y_int, y_pred, n_clases=3)
    resumen = imprimir_resumen("Multiclase", accuracy, aciertos, len(y_int), errores, tiempo)

    # 5. Visualización
    Visualizador(modo='multiclase').graficar(
        X                  = X,
        y_real_int         = y_int,
        y_pred_int         = y_pred,
        historial_perdida  = red.historial_perdida,
        historial_accuracy = red.historial_accuracy,
        titulo             = cfg['titulo'],
        ruta_salida        = cfg['ruta_salida'],
    )

    return resumen


# ─────────────────────────────────────────────────────
# COMPARATIVA FINAL
# ─────────────────────────────────────────────────────

def imprimir_comparativa(resultados: list[dict]) -> None:
    separador("COMPARATIVA FINAL")
    print(f"\n  {'Escenario':<14} {'Accuracy':>10} {'Aciertos':>10} {'Errores':>10} {'Tiempo':>10}")
    print("  " + "-" * 50)
    for r in resultados:
        print(
            f"  {r['nombre']:<14}"
            f"  {r['accuracy']:>8.2f}%"
            f"  {r['aciertos']:>9}"
            f"  {r['errores']:>9}"
            f"  {r['tiempo']:>8.2f}s"
        )
    print()


# ─────────────────────────────────────────────────────
# ARGUMENTOS DE LÍNEA DE COMANDOS
# ─────────────────────────────────────────────────────

def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Red Neuronal — Clasificación 3D (Binario y Multiclase)"
    )
    parser.add_argument('--s1', action='store_true', help='Correr solo Escenario 1 (binario)')
    parser.add_argument('--s2', action='store_true', help='Correr solo Escenario 2 (multiclase)')
    return parser.parse_args()


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main() -> None:
    args = parsear_args()
    correr_s1 = args.s1 or (not args.s1 and not args.s2)
    correr_s2 = args.s2 or (not args.s1 and not args.s2)

    resultados = []

    if correr_s1:
        resultados.append(correr_escenario1())

    if correr_s2:
        resultados.append(correr_escenario2())

    if len(resultados) == 2:
        imprimir_comparativa(resultados)


if __name__ == '__main__':
    main()