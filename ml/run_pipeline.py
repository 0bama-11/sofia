"""
Script de conveniencia para ejecutar el pipeline completo:
  1. Descargar dataset Food-101
  2. Entrenar modelo
  3. Exportar para inferencia

Uso:
    python -m ml.run_pipeline
    python -m ml.run_pipeline --epochs 15 --batch-size 64
    python -m ml.run_pipeline --skip-download
    python -m ml.run_pipeline --resume auto --epochs 10
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo ML")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-download", action="store_true",
                        help="Saltar descarga si ya tienes el dataset")
    parser.add_argument("--skip-train", action="store_true",
                        help="Saltar entrenamiento (solo exportar)")
    parser.add_argument("--resume", type=str, default=None,
                        help='Continuar entrenamiento desde checkpoint. '
                             'Usa "auto" para best_model.pt.')
    args = parser.parse_args()

    # ─── 1. Download ───
    if not args.skip_download:
        print("=" * 60)
        print("PASO 1: Descargando dataset Food-101")
        print("=" * 60)
        from ml.download_dataset import download_food101
        download_food101()
        print()
    else:
        print("Descarga saltada (--skip-download)")

    # ─── 2. Train ───
    if not args.skip_train:
        print("=" * 60)
        print("PASO 2: Entrenando modelo")
        print("=" * 60)
        from ml.train import train
        train_args = argparse.Namespace(
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        train(train_args)
        print()
    else:
        print("Entrenamiento saltado (--skip-train)")

    # ─── 3. Export ───
    print("=" * 60)
    print("PASO 3: Exportando modelo")
    print("=" * 60)
    from ml.export_model import export_model
    export_args = argparse.Namespace(model=None)
    export_model(export_args)

    print()
    print("=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print("Ahora ejecuta la app con: python main.py")
    print("El modelo real será cargado automáticamente.")


if __name__ == "__main__":
    main()
