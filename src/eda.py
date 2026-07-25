"""
Análisis exploratorio de datos (EDA) para el dataset de desintegración radiactiva.

Genera:
1. Gráfico de actividad vs. tiempo (escala lineal), con barras de error (sigma)
2. Gráfico en escala semi-logarítmica (ln(actividad) vs. tiempo) — en una
   desintegración exponencial pura, esto debe verse como una línea recta,
   lo cual es una excelente verificación visual del modelo.
3. Estadísticas descriptivas básicas del dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(path="data/decay_data.csv"):
    df = pd.read_csv(path)
    return df


def print_summary(df):
    print("=" * 50)
    print("RESUMEN ESTADÍSTICO")
    print("=" * 50)
    print(df[["time", "activity", "sigma"]].describe())
    print()
    print(f"Actividad inicial medida (t≈0): {df['activity'].iloc[0]:.2f}")
    print(f"Actividad final medida (t={df['time'].iloc[-1]:.1f}): {df['activity'].iloc[-1]:.2f}")
    print()


def plot_linear(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        df["time"], df["activity"], yerr=df["sigma"],
        fmt="o", color="steelblue", ecolor="lightgray",
        elinewidth=1.5, capsize=3, markersize=5,
        label="Datos medidos (± σ)"
    )
    ax.plot(
        df["time"], df["activity_true"],
        color="firebrick", linestyle="--", linewidth=1.5,
        label="Señal verdadera (sin ruido)"
    )

    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Actividad")
    ax.set_title("Actividad vs. Tiempo — Escala lineal")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {output_path}")


def plot_semilog(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Solo graficamos puntos con actividad > 0 (log no está definido en 0)
    mask = df["activity"] > 0
    ax.errorbar(
        df["time"][mask], df["activity"][mask],
        yerr=df["sigma"][mask],
        fmt="o", color="steelblue", ecolor="lightgray",
        elinewidth=1.5, capsize=3, markersize=5,
        label="Datos medidos"
    )
    ax.set_yscale("log")

    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Actividad (escala log)")
    ax.set_title("Actividad vs. Tiempo — Escala semi-logarítmica")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {output_path}")


def main():
    df = load_data()
    print_summary(df)

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    plot_linear(df, output_dir / "eda_linear.png")
    plot_semilog(df, output_dir / "eda_semilog.png")


if __name__ == "__main__":
    main()