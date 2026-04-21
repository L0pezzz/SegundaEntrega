import numpy as np
from skimage import measure   # Marching Cubes


def calcular_frontera_binaria(
    red,
    rango_min: np.ndarray,
    rango_max: np.ndarray,
    resolucion: int = 30,
) -> dict:
    """
    Calcula la frontera de decisión P(clase=1) = 0.5 para clasificación binaria.

    Args:
        red        : instancia de MLP ya entrenada (o con pesos de alguna época)
        rango_min  : array shape (3,) — mínimo por dimensión
        rango_max  : array shape (3,) — máximo por dimensión
        resolucion : número de divisiones por eje (30 → 27000 puntos)

    Returns:
        dict con claves:
            'vertices'  : lista de [x, y, z] en coordenadas del espacio de datos
            'triangulos': lista de [i, j, k] — índices de vértices
        Si la frontera no existe (accuracy perfecta), retorna listas vacías.
    """
    # Crear grilla 3D
    xs = np.linspace(rango_min[0], rango_max[0], resolucion)
    ys = np.linspace(rango_min[1], rango_max[1], resolucion)
    zs = np.linspace(rango_min[2], rango_max[2], resolucion)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Predicciones de probabilidad
    probs = red.predecir_proba(grid)          # shape (N, 1)
    volumen = probs.reshape(resolucion, resolucion, resolucion)

    # Verificar que el volumen tenga valores a ambos lados del umbral 0.5
    if volumen.max() < 0.5 or volumen.min() > 0.5:
        return {'vertices': [], 'triangulos': []}

    # Marching Cubes: encuentra la superficie donde volumen == 0.5
    verts_idx, faces, _, _ = measure.marching_cubes(volumen, level=0.5)

    # Los vértices están en índices de grilla (0..resolucion-1)
    # Convertir a coordenadas del espacio de datos
    escala = (rango_max - rango_min) / (resolucion - 1)
    verts_mundo = verts_idx * escala + rango_min

    return {
        'vertices':   verts_mundo.tolist(),
        'triangulos': faces.tolist(),
    }


def calcular_fronteras_multiclase(
    red,
    rango_min: np.ndarray,
    rango_max: np.ndarray,
    resolucion: int = 25,
) -> list[dict]:
    """
    Para cada clase k, calcula la región donde P(clase=k) es máxima.
    Retorna una lista con 3 fronteras, una por clase.

    Args:
        red        : instancia de MLP multiclase
        rango_min  : array shape (3,)
        rango_max  : array shape (3,)
        resolucion : divisiones por eje

    Returns:
        Lista de 3 dicts, cada uno con 'vertices' y 'triangulos'.
        La clase k está en la posición k de la lista.
    """
    xs = np.linspace(rango_min[0], rango_max[0], resolucion)
    ys = np.linspace(rango_min[1], rango_max[1], resolucion)
    zs = np.linspace(rango_min[2], rango_max[2], resolucion)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # shape (N, 3)
    probs = red.predecir_proba(grid)
    # Para cada clase k, el volumen es P(clase=k)
    escala = (rango_max - rango_min) / (resolucion - 1)

    fronteras = []
    for k in range(3):
        vol = probs[:, k].reshape(resolucion, resolucion, resolucion)

        if vol.max() < 0.5 or vol.min() > 0.5:
            fronteras.append({'vertices': [], 'triangulos': []})
            continue

        verts_idx, faces, _, _ = measure.marching_cubes(vol, level=0.5)
        verts_mundo = verts_idx * escala + rango_min

        fronteras.append({
            'vertices':   verts_mundo.tolist(),
            'triangulos': faces.tolist(),
        })

    return fronteras