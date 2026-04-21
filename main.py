import argparse
import time
import numpy as np

from datos      import GeneradorBinario, GeneradorMulticlase
from modelo     import MLP
from exportador import exportar


# ─── Parseo de argumentos ────────────────────────────────────────────────────

def parsear_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLP 3D — Clasificación con exportación a Unity"
    )
    p.add_argument('--escenario',     type=int,   required=True,
                   choices=[1, 2],
                   help='1=binario [3→4→1]  2=multiclase [3→8→3]')
    p.add_argument('--distribucion',  type=str,   default='normal',
                   choices=['normal', 'exponencial', 'laplace', 'uniforme'])
    p.add_argument('--dispersion',    type=float, default=0.8,
                   help='Dispersión de las nubes (equivale a sigma en distribución normal)')
    p.add_argument('--epocas',        type=int,   default=300)
    p.add_argument('--puntos',        type=int,   default=200,
                   help='Puntos POR CLASE (total = puntos × n_clases)')
    p.add_argument('--lr',            type=float, default=0.05,
                   help='Learning rate')
    p.add_argument('--batch',         type=int,   default=32,
                   help='Tamaño del mini-batch')
    p.add_argument('--salida',        type=str,   default='resultado.json')
    p.add_argument('--intervalo',     type=int,   default=5,
                   help='Guardar frontera de decisión cada N épocas')
    return p.parse_args()


# ─── Escenario 1 — Binario ────────────────────────────────────────────────────

def correr_escenario1(args: argparse.Namespace) -> None:
    print("\n" + "="*55)
    print("  ESCENARIO 1 — Clasificación Binaria  [3 → 4 → 1]")
    print("="*55)

    n_total = args.puntos * 2  # 2 clases
    gen = GeneradorBinario(
        n=n_total,
        ruido=args.dispersion,
        semilla=42,
        distribucion=args.distribucion,
    )
    gen.generar()
    gen.resumen()
    X, y = gen.normalizar()
    y_int = y.flatten().astype(int)

    np.random.seed(42)
    red = MLP(
        capas=[3, 4, 1],
        modo='binario',
        lr=args.lr,
        epocas=args.epocas,
        batch_size=args.batch,
    )
    red.resumen()

    print("\n  Entrenando...")
    t0 = time.time()
    pesos_por_epoca = red.entrenar(X, y, verbose=True, guardar_pesos=True)
    tiempo = time.time() - t0

    y_pred   = red.predecir(X)
    aciertos = int((y_pred == y_int).sum())
    accuracy = aciertos / len(y_int) * 100

    print(f"\n  Accuracy final : {accuracy:.2f}%")
    print(f"  Tiempo         : {tiempo:.3f} s")

    metadata = {
        'escenario':             1,
        'modo':                  'binario',
        'capas':                 [3, 4, 1],
        'distribucion':          args.distribucion,
        'dispersion':            args.dispersion,
        'epocas':                args.epocas,
        'puntos_por_clase':      args.puntos,
        'lr':                    args.lr,
        'batch_size':            args.batch,
        'tiempo_entrenamiento_s': round(tiempo, 4),
        'accuracy_final':        round(accuracy, 2),
        'n_clases':              2,
    }

    exportar(
        ruta=args.salida,
        red=red,
        X=X,
        y_int=y_int,
        metadata=metadata,
        pesos_por_epoca=pesos_por_epoca,
        intervalo_frontera=args.intervalo,
    )


# ─── Escenario 2 — Multiclase ─────────────────────────────────────────────────

def correr_escenario2(args: argparse.Namespace) -> None:
    print("\n" + "="*55)
    print("  ESCENARIO 2 — Clasificación Multiclase  [3 → 8 → 3]")
    print("="*55)

    n_total = args.puntos * 3  # 3 clases
    gen = GeneradorMulticlase(
        n=n_total,
        ruido=args.dispersion,
        semilla=42,
        distribucion=args.distribucion,
    )
    gen.generar()
    gen.resumen()
    X, y_oh, y_int = gen.normalizar()

    np.random.seed(42)
    red = MLP(
        capas=[3, 8, 3],
        modo='multiclase',
        lr=args.lr,
        epocas=args.epocas,
        batch_size=args.batch,
    )
    red.resumen()

    print("\n  Entrenando...")
    t0 = time.time()
    pesos_por_epoca = red.entrenar(X, y_oh, verbose=True, guardar_pesos=True)
    tiempo = time.time() - t0

    y_pred   = red.predecir(X)
    aciertos = int((y_pred == y_int).sum())
    accuracy = aciertos / len(y_int) * 100

    print(f"\n  Accuracy final : {accuracy:.2f}%")
    print(f"  Tiempo         : {tiempo:.3f} s")

    metadata = {
        'escenario':              2,
        'modo':                   'multiclase',
        'capas':                  [3, 8, 3],
        'distribucion':           args.distribucion,
        'dispersion':             args.dispersion,
        'epocas':                 args.epocas,
        'puntos_por_clase':       args.puntos,
        'lr':                     args.lr,
        'batch_size':             args.batch,
        'tiempo_entrenamiento_s': round(tiempo, 4),
        'accuracy_final':         round(accuracy, 2),
        'n_clases':               3,
    }

    exportar(
        ruta=args.salida,
        red=red,
        X=X,
        y_int=y_int,
        metadata=metadata,
        pesos_por_epoca=pesos_por_epoca,
        intervalo_frontera=args.intervalo,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parsear_args()
    if args.escenario == 1:
        correr_escenario1(args)
    else:
        correr_escenario2(args)
    print("\n  ¡Listo! Unity puede cargar el JSON ahora.\n")