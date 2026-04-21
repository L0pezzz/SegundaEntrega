import json
import copy
import numpy as np

from frontera import calcular_frontera_binaria, calcular_fronteras_multiclase


def _copiar_pesos(red) -> list[dict]:
    """
    Crea una copia profunda de los pesos actuales del modelo.
    Necesario para guardar el estado de cada época.
    """
    return [
        {
            'W': W.tolist(),
            'b': b.tolist(),
        }
        for W, b in zip(red.W, red.b)
    ]


def _restaurar_pesos(red, pesos: list[dict]) -> None:
    """Carga un conjunto de pesos guardados de vuelta en el modelo."""
    for l, capa in enumerate(pesos):
        red.W[l] = np.array(capa['W'])
        red.b[l] = np.array(capa['b'])


def exportar(
    ruta: str,
    red,                         # instancia de MLP ya entrenada
    X: np.ndarray,               # datos normalizados (N, 3)
    y_int: np.ndarray,           # etiquetas enteras (N,) — 0,1 o 0,1,2
    metadata: dict,              # parámetros de configuración
    pesos_por_epoca: list[list[dict]],  # pesos guardados en cada época
    intervalo_frontera: int = 5, # guardar frontera cada N épocas
    resolucion_frontera: int = 28,
) -> None:
    """
    Escribe resultado.json en la ruta indicada.

    Args:
        ruta               : path del archivo de salida, ej. 'resultado.json'
        red                : MLP entrenado
        X                  : datos de entrada normalizados
        y_int              : clases enteras de cada muestra
        metadata           : dict con configuración y métricas
        pesos_por_epoca    : lista de listas — pesos al final de cada época
        intervalo_frontera : cada cuántas épocas calcular la malla 3D
        resolucion_frontera: resolución de la grilla para Marching Cubes
    """
    print("\n  Calculando fronteras de decisión por época...")

    modo = red.modo
    rango_min = X.min(axis=0) - 0.5
    rango_max = X.max(axis=0) + 0.5

    fronteras_json = []
    n_epocas = len(pesos_por_epoca)

    for ep_idx in range(0, n_epocas, intervalo_frontera):
        epoca = ep_idx + 1  # épocas van de 1 a N
        _restaurar_pesos(red, pesos_por_epoca[ep_idx])

        if modo == 'binario':
            malla = calcular_frontera_binaria(
                red, rango_min, rango_max, resolucion=resolucion_frontera
            )
            fronteras_json.append({'epoca': epoca, 'clases': [malla]})
        else:
            mallas = calcular_fronteras_multiclase(
                red, rango_min, rango_max, resolucion=resolucion_frontera
            )
            fronteras_json.append({'epoca': epoca, 'clases': mallas})

        porcentaje = (ep_idx + 1) / n_epocas * 100
        print(f"\r    {porcentaje:.0f}%  (época {epoca}/{n_epocas})", end='', flush=True)

    # Restaurar pesos finales
    _restaurar_pesos(red, pesos_por_epoca[-1])
    print("\n  ✓ Fronteras calculadas.")

    # Construir objeto JSON
    salida = {
        'metadata': metadata,
        'datos': {
            'X':      X.tolist(),
            'y_int':  y_int.tolist(),
        },
        'historial': {
            'perdida':   red.historial_perdida,
            'accuracy': [round(a * 100, 2) for a in red.historial_accuracy],
        },
        'fronteras': fronteras_json,
    }

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(salida, f, separators=(',', ':'))  # compacto, sin espacios

    import os
    tam_mb = os.path.getsize(ruta) / 1024 / 1024
    print(f"  ✓ Guardado: {ruta}  ({tam_mb:.2f} MB)")