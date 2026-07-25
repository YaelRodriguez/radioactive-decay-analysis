"""
Paso 5: Cálculo de la vida media (t_1/2) con propagación de incertidumbre.

t_half = ln(2) / lambda

Nota: t_half depende únicamente de lambda, por lo que el término de
covarianza cruzada A0-lambda (cov_A0_lambda) no contribuye: su derivada
parcial respecto a A0 es cero. Solo se necesita la varianza de lambda
(lambda_err al cuadrado).
"""

import json
import numpy as np

RESULTS_PATH = "results/fit_results.json"

def main():
    with open(RESULTS_PATH, "r") as f:
        fit = json.load(f)

    lam = fit["lambda_fit"]
    sigma_lambda = fit["lambda_err"]

    t_half = np.log(2) / lam
    sigma_t_half = (np.log(2) / lam**2) * sigma_lambda

    fit["t_half"] = t_half
    fit["t_half_uncertainty"] = sigma_t_half

    with open(RESULTS_PATH, "w") as f:
        json.dump(fit, f, indent=2)

    print(f"Vida media: t_1/2 = {t_half:.4f} ± {sigma_t_half:.4f} (unidades de tiempo)")

if __name__ == "__main__":
    main()