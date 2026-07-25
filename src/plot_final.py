"""
Paso 6: Visualizacion final del ajuste exponencial.
Muestra los datos originales, la curva ajustada, y una banda de
incertidumbre de 1-sigma propagada desde la covarianza completa de A0 y lambda.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/decay_data.csv"
RESULTS_PATH = "results/fit_results.json"
OUTPUT_PATH = "figures/final_fit.png"


def decay_model(t, A0, lam):
    return A0 * np.exp(-lam * t)


def decay_model_variance(t, A0, lam, A0_err, lambda_err, cov_A0_lambda):
    """
    Propagacion de incertidumbre para N(t) = A0 * exp(-lambda * t).

    dN/dA0 = exp(-lambda * t)
    dN/dlambda = -A0 * t * exp(-lambda * t)

    sigma_N^2 = (dN/dA0)^2 * var(A0) + (dN/dlambda)^2 * var(lambda)
                + 2 * (dN/dA0) * (dN/dlambda) * cov(A0, lambda)
    """
    dN_dA0 = np.exp(-lam * t)
    dN_dlam = -A0 * t * np.exp(-lam * t)

    var_N = (dN_dA0 ** 2) * (A0_err ** 2) \
        + (dN_dlam ** 2) * (lambda_err ** 2) \
        + 2 * dN_dA0 * dN_dlam * cov_A0_lambda

    return var_N


def main():
    df = pd.read_csv(DATA_PATH)
    with open(RESULTS_PATH, "r") as f:
        fit = json.load(f)

    A0 = fit["A0_fit"]
    lam = fit["lambda_fit"]
    A0_err = fit["A0_err"]
    lambda_err = fit["lambda_err"]
    cov_A0_lambda = fit["cov_A0_lambda"]
    t_half = fit["t_half"]

    t_col = df.columns[0]
    n_col = df.columns[1]

    t_fine = np.linspace(df[t_col].min(), df[t_col].max(), 300)
    n_fit = decay_model(t_fine, A0, lam)

    var_n = decay_model_variance(t_fine, A0, lam, A0_err, lambda_err, cov_A0_lambda)
    sigma_n = np.sqrt(var_n)

    plt.figure(figsize=(9, 6))
    plt.scatter(df[t_col], df[n_col], s=15, alpha=0.6, label="Datos experimentales", color="steelblue")
    plt.plot(t_fine, n_fit, color="crimson", lw=2, label=f"Ajuste exponencial")
    plt.fill_between(t_fine, n_fit - sigma_n, n_fit + sigma_n,
                      color="crimson", alpha=0.2, label="Banda de incertidumbre (1σ)")

    plt.axvline(t_half, color="gray", ls="--", lw=1,
                label=f"t½ = {t_half:.2f} ± {fit['t_half_uncertainty']:.2f}")

    plt.xlabel(t_col)
    plt.ylabel(n_col)
    plt.title("Ajuste de decaimiento radiactivo con banda de incertidumbre")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Figura guardada en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()