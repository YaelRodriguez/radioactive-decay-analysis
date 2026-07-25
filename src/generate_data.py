"""
Genera datos sintéticos de desintegración radiactiva con ruido gaussiano realista.

Modelo físico: A(t) = A0 * exp(-lambda * t)
donde:
    A0     = actividad inicial
    lambda = constante de desintegración (lambda = ln(2) / t_half)
    t_half = vida media (half-life)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- Parámetros "verdaderos" del proceso simulado ---
A0_TRUE = 1000.0        # actividad inicial (cuentas/s)
T_HALF_TRUE = 12.5      # vida media en unidades de tiempo (ej. días)
LAMBDA_TRUE = np.log(2) / T_HALF_TRUE

# --- Parámetros de muestreo ---
T_MAX = 60.0            # tiempo total de observación
N_POINTS = 40            # número de mediciones
NOISE_LEVEL = 0.05        # 5% de ruido relativo (heterocedástico, típico en conteo radiactivo)
SEED = 42                # para reproducibilidad

def generate_decay_data(
    a0=A0_TRUE,
    decay_lambda=LAMBDA_TRUE,
    t_max=T_MAX,
    n_points=N_POINTS,
    noise_level=NOISE_LEVEL,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    # Tiempos de medición (no necesariamente equiespaciados sería más realista,
    # pero usamos espaciado uniforme para simplicidad del EDA)
    t = np.linspace(0, t_max, n_points)

    # Señal "verdadera" sin ruido
    activity_true = a0 * np.exp(-decay_lambda * t)

    # Ruido gaussiano proporcional a la señal (simula ruido de conteo tipo Poisson
    # aproximado por gaussiano para actividades altas) + un piso de ruido de fondo
    sigma = noise_level * activity_true + 1.0  # +1 evita sigma=0 en colas
    noise = rng.normal(loc=0.0, scale=sigma)

    activity_measured = activity_true + noise
    activity_measured = np.clip(activity_measured, a_min=0, a_max=None)  # la actividad no puede ser negativa

    df = pd.DataFrame({
        "time": t,
        "activity": activity_measured,
        "activity_true": activity_true,   # útil para validar el ajuste después
        "sigma": sigma,                    # incertidumbre por punto, útil para curve_fit(sigma=...)
    })

    return df


def main():
    df = generate_decay_data()

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "decay_data.csv"

    df.to_csv(output_path, index=False)
    print(f"Datos generados: {len(df)} puntos")
    print(f"Parámetros verdaderos -> A0={A0_TRUE}, t_half={T_HALF_TRUE}, lambda={LAMBDA_TRUE:.5f}")
    print(f"Guardado en: {output_path}")


if __name__ == "__main__":
    main()