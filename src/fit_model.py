"""
Ajuste del modelo de desintegración exponencial a los datos medidos,
usando scipy.optimize.curve_fit.

Modelo: A(t) = A0 * exp(-lambda * t)

Usamos las incertidumbres (sigma) de cada punto como pesos del ajuste
(least squares ponderado), lo cual es más riguroso que un ajuste sin pesos:
los puntos con menor incertidumbre pesan más en la determinación de los
parámetros óptimos.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import json


def decay_model(t, A0, decay_lambda):
    """Modelo de desintegración exponencial."""
    return A0 * np.exp(-decay_lambda * t)


def load_data(path="data/decay_data.csv"):
    return pd.read_csv(path)


def fit_decay(df):
    t = df["time"].values
    activity = df["activity"].values
    sigma = df["sigma"].values

    # Estimados iniciales razonables (importante para que curve_fit converja bien)
    p0 = [activity[0], 0.05]

    popt, pcov = curve_fit(
        decay_model, t, activity,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,  # sigma representa incertidumbres reales, no solo pesos relativos
    )

    A0_fit, lambda_fit = popt
    # Incertidumbres = raíz cuadrada de la diagonal de la matriz de covarianza
    A0_err, lambda_err = np.sqrt(np.diag(pcov))

    return {
        "A0_fit": A0_fit,
        "A0_err": A0_err,
        "lambda_fit": lambda_fit,
        "lambda_err": lambda_err,
        "pcov": pcov,
    }


def compute_goodness_of_fit(df, result):
    t = df["time"].values
    activity = df["activity"].values
    sigma = df["sigma"].values

    predicted = decay_model(t, result["A0_fit"], result["lambda_fit"])
    residuals = activity - predicted

    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(t) - 2  # 2 parámetros ajustados
    chi2_reduced = chi2 / dof

    return {"chi2": chi2, "dof": dof, "chi2_reduced": chi2_reduced}


def main():
    df = load_data()
    result = fit_decay(df)
    gof = compute_goodness_of_fit(df, result)

    print("=" * 50)
    print("RESULTADOS DEL AJUSTE")
    print("=" * 50)
    print(f"A0     = {result['A0_fit']:.2f} ± {result['A0_err']:.2f}")
    print(f"lambda = {result['lambda_fit']:.5f} ± {result['lambda_err']:.5f}")
    print()
    print(f"Chi-cuadrado         = {gof['chi2']:.2f}")
    print(f"Grados de libertad   = {gof['dof']}")
    print(f"Chi-cuadrado reducido = {gof['chi2_reduced']:.3f}  (ideal ≈ 1.0)")
    print()

    # Guardamos los resultados para usarlos en pasos siguientes (vida media, gráficos)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = {
        "A0_fit": result["A0_fit"],
        "A0_err": result["A0_err"],
        "lambda_fit": result["lambda_fit"],
        "lambda_err": result["lambda_err"],
        "chi2": gof["chi2"],
        "dof": gof["dof"],
        "chi2_reduced": gof["chi2_reduced"],
        "cov_A0_lambda": result["pcov"][0, 1],  # covarianza cruzada, útil para propagación de errores
    }

    output_path = output_dir / "fit_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()