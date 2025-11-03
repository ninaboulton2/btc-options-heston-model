"""
Module de simulation et pricing pour le modèle de Heston.

Dynamiques sous mesure risque-neutre :
    dS_t = r S_t dt + sqrt(v_t) S_t dW_1(t)
    dv_t = κ(θ - v_t) dt + ξ sqrt(v_t) dW_2(t)

avec ρ = corr(dW_1, dW_2). Discrétisation par Euler-Maruyama avec réflexion.
"""

from __future__ import annotations

import numpy as np


def simulate_heston_paths(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    N: int,
    n_paths: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simule des trajectoires du modèle de Heston.

    Returns
    -------
    S : np.ndarray
        Trajectoires du prix (n_paths, N+1).
    v : np.ndarray
        Trajectoires de la variance (n_paths, N+1).
    """
    rng = np.random.default_rng(random_state)
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    
    S = np.empty((n_paths, N + 1))
    v = np.empty((n_paths, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Facteur de Cholesky pour la corrélation
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho ** 2)]])
    
    for i in range(N):
        z = rng.standard_normal(size=(n_paths, 2))
        dW = z @ L.T
        dW1 = dW[:, 0] * sqrt_dt
        dW2 = dW[:, 1] * sqrt_dt
        
        v_prev = v[:, i]
        v_new = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(np.maximum(v_prev, 0.0)) * dW2
        v[:, i + 1] = np.maximum(v_new, 0.0)  # Réflexion pour positivité
        
        S[:, i + 1] = S[:, i] * np.exp(
            (r - 0.5 * np.maximum(v_prev, 0.0)) * dt + np.sqrt(np.maximum(v_prev, 0.0)) * dW1
        )
    
    return S, v


def european_call_mc(
    S0: float,
    K: float,
    r: float,
    T: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    n_paths: int = 10000,
    N: int = 200,
    random_state: int | None = None,
) -> float:
    """Estime le prix d'un call européen par Monte-Carlo sous le modèle de Heston.

    Returns
    -------
    price : float
        Prix du call estimé par Monte-Carlo.
    """
    S_paths, _ = simulate_heston_paths(
        S0, v0, r, kappa, theta, xi, rho, T, N, n_paths, random_state
    )
    payoffs = np.maximum(S_paths[:, -1] - K, 0.0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price