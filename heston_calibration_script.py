"""
Script de calibration du modèle de Heston sur des données d’options Bitcoin.

Ce script charge un fichier `data.csv` contenant un snapshot des options BTC
de Deribit, extrait les options de type call, convertit les dates d’échéance
et les strikes, calcule le prix observé en dollars et prépare un échantillon de
contrats pour la calibration.  Il utilise ensuite le module `heston_model.py`
pour simuler des trajectoires selon le modèle de Heston et calcule les prix
théoriques par Monte‑Carlo.  Les paramètres du modèle sont ajustés par
minimisation de la somme des carrés entre les prix de marché et les prix
théoriques.  Enfin, il compare les prix et volatilités implicites observés
et modélisés.

"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import heston_model


def bs_call_price(S0: float, K: float, r: float, T: float, sigma: float) -> float:
    """Prix d'un call européen par la formule de Black-Scholes."""
    if sigma <= 0 or T <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price: float, S0: float, K: float, r: float, T: float) -> float:
    """Volatilité implicite par inversion binaire de Black-Scholes."""
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-12:
        return 0.0
    low, high = 1e-8, 5.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        v = bs_call_price(S0, K, r, T, mid)
        if v > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def parse_instrument_name(name: str):
    """Extrait strike et date d'échéance du nom d'instrument.

    Format: 'BTC-DDMMMYY-STRIKE-C' ou 'BTC-DDMMMYY-STRIKE-P'.
    Returns (strike, expiry) ou (None, None).
    """
    pattern = re.compile(r"BTC-(\d{2}[A-Z]{3}\d{2})-(\d+)-(C|P)")
    m = pattern.match(name)
    if not m:
        return None, None
    date_str, strike_str, _ = m.groups()
    strike = float(strike_str)
    expiry = datetime.strptime(date_str, "%d%b%y")
    return strike, expiry


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Charge et prépare les données pour la calibration.

    Extrait les options call, calcule maturité, prix USD et volatilité implicite.
    """
    df = pd.read_csv(csv_path)
    calls = df[df["instrument_name"].str.endswith("C")].copy()

    strikes, expiries = zip(*calls["instrument_name"].apply(parse_instrument_name))
    calls["strike"] = strikes
    calls["expiry"] = expiries

    calls["timestamp_dt"] = pd.to_datetime(calls["timestamp"], unit="ms")
    calls["T"] = (
        (calls["expiry"] - calls["timestamp_dt"]).dt.total_seconds() /
        (365.0 * 24.0 * 3600.0)
    )

    calls = calls[(calls["T"] > 0) & calls["strike"].notnull()]
    calls["iv_observed"] = calls["mark_iv"] / 100.0
    calls["price_usd"] = calls["mark_price"] * calls["underlying_price"]
    return calls[[
        "instrument_name", "strike", "T", "price_usd",
        "underlying_price", "iv_observed"
    ]].reset_index(drop=True)


def filter_homogeneous_options(data: pd.DataFrame, min_maturity: float = 0.02,
                                min_moneyness: float = 0.8, max_moneyness: float = 1.2) -> pd.DataFrame:
    """Filtre les options selon maturité et moneyness."""
    data = data.copy()
    data["moneyness"] = data["strike"] / data["underlying_price"]
    filtered = data[
        (data["T"] >= min_maturity) &
        (data["moneyness"] >= min_moneyness) &
        (data["moneyness"] <= max_moneyness)
    ].copy()
    
    return filtered.reset_index(drop=True)


def select_homogeneous_options(data: pd.DataFrame, target_maturities: list,
                                moneyness_tolerance: float = 0.15,
                                strikes_per_maturity: int = 4,
                                random_state: int = 42) -> pd.DataFrame:
    """Sélectionne des options autour de maturités cibles proches de la monnaie."""
    rng = np.random.default_rng(random_state)
    samples = []
    used_indices = set()
    
    for target_T in target_maturities:
        moneyness_mask = (data["moneyness"] >= 1.0 - moneyness_tolerance) & \
                        (data["moneyness"] <= 1.0 + moneyness_tolerance)
        
        candidates = data[moneyness_mask].copy()
        candidates = candidates[~candidates.index.isin(used_indices)]
        
        if len(candidates) == 0:
            print(f"Avertissement : Aucune option trouvée pour la maturité cible {target_T:.3f} ans "
                  f"(≈{target_T*12:.1f} mois)")
            continue
        
        candidates["maturity_distance"] = abs(candidates["T"] - target_T)
        candidates["moneyness_distance"] = abs(candidates["moneyness"] - 1.0)
        candidates = candidates.sort_values(["maturity_distance", "moneyness_distance"])
        
        n = min(strikes_per_maturity, len(candidates))
        selected = candidates.head(n)
        used_indices.update(selected.index)
        samples.append(selected)
        
        print(f"Maturité cible {target_T:.3f} ans (≈{target_T*12:.1f} mois): "
              f"{len(selected)} options sélectionnées "
              f"(T moyen: {selected['T'].mean():.3f} ans, "
              f"moneyness: {selected['moneyness'].min():.3f} - {selected['moneyness'].max():.3f})")
    
    if samples:
        result = pd.concat(samples).reset_index(drop=True)
        result = result.drop(columns=["moneyness_distance", "maturity_distance"], errors="ignore")
        return result
    
    return pd.DataFrame(columns=data.columns)


def calibrate_heston(obs_df: pd.DataFrame, risk_free_rate: float = 0.02,
                     n_paths: int = 5000, n_steps: int = 100,
                     maxiter: int = 200, n_initial_points: int = 5,
                     verbose: bool = True) -> tuple:
    """Calibre le modèle de Heston par minimisation des erreurs quadratiques.

    Returns
    -------
    best_result : OptimizeResult
        Meilleur résultat de l'optimisation.
    """
    # Bornes : [kappa, theta, xi, rho, v0]
    bounds = (
        [0.5, 0.1, 0.05, -0.9, 0.01],
        [15.0, 0.8, 1.0, 0.0, 1.0]
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        kappa, theta, xi, rho, v0 = params
        res = []
        for _, row in obs_df.iterrows():
            model_price = heston_model.european_call_mc(
                S0=row["underlying_price"],
                K=row["strike"],
                r=risk_free_rate,
                T=row["T"],
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                v0=v0,
                n_paths=n_paths,
                N=n_steps,
                random_state=42,
            )
            residual = (model_price - row["price_usd"]) / row["price_usd"]
            res.append(residual)
        return np.array(res)

    np.random.seed(42)
    initial_points = []
    initial_points.append([2.0, 0.2, 0.3, -0.5, 0.2])
    
    for i in range(n_initial_points - 1):
        kappa = np.random.uniform(0.5, 15.0)
        theta = np.random.uniform(0.1, 0.8)
        xi = np.random.uniform(0.1, 1.0)
        rho = np.random.uniform(-0.9, 0.0)
        v0 = np.random.uniform(0.02, 0.8)
        initial_points.append([kappa, theta, xi, rho, v0])
    
    best_result = None
    best_cost = np.inf
    
    if verbose:
        print(f"Test de {n_initial_points} points de départ différents...")
    
    for i, x0 in enumerate(initial_points):
        if verbose:
            print(f"\nPoint de départ {i+1}/{n_initial_points}: "
                  f"κ={x0[0]:.2f}, θ={x0[1]:.4f}, ξ={x0[2]:.3f}, ρ={x0[3]:.3f}, v₀={x0[4]:.4f}")
        
        try:
            res = least_squares(
                residuals,
                x0=np.array(x0),
                bounds=bounds,
                method='trf',
                max_nfev=maxiter,
                verbose=0,
            )
            
            cost = np.sum(res.fun ** 2)
            
            if verbose:
                print(f"  → Coût final: {cost:.6f}")
            
            if cost < best_cost:
                best_cost = cost
                best_result = res
                if verbose:
                    print(f"  → ✓ Nouveau meilleur résultat!")
                    
        except Exception as e:
            if verbose:
                print(f"  → Échec de l'optimisation: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("Aucune optimisation n'a réussi. Vérifiez les données et les paramètres.")
    
    if verbose:
        print(f"\n✓ Meilleur résultat trouvé avec coût: {best_cost:.6f}")
    
    return best_result


def plot_volatility_analysis(sample_df: pd.DataFrame, calibrated_params: np.ndarray,
                              underlying_price: float, risk_free_rate: float = 0.02,
                              n_paths: int = 10000, n_steps: int = 100):
    """Génère toutes les visualisations pour l'analyse de la surface de volatilité."""
    kappa, theta, xi, rho, v0 = calibrated_params
    
    import os
    os.makedirs("volatility_analysis_plots", exist_ok=True)
    
    print("\n" + "="*80)
    print("GÉNÉRATION DES GRAPHIQUES D'ANALYSE DE LA SURFACE DE VOLATILITÉ")
    print("="*80)
    
    print("\n1. Génération de la surface de volatilité implicite...")
    fig = plt.figure(figsize=(16, 6))
    
    # Surface observée
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(sample_df["moneyness"], sample_df["T"], sample_df["iv_observed"],
                          c=sample_df["iv_observed"], cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('Moneyness (K/S₀)')
    ax1.set_ylabel('Maturité (années)')
    ax1.set_zlabel('Volatilité implicite')
    ax1.set_title('Surface observée (marché)')
    plt.colorbar(scatter1, ax=ax1)
    
    # Surface modélisée
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(sample_df["moneyness"], sample_df["T"], sample_df["iv_model"],
                          c=sample_df["iv_model"], cmap='viridis', s=50, alpha=0.7)
    ax2.set_xlabel('Moneyness (K/S₀)')
    ax2.set_ylabel('Maturité (années)')
    ax2.set_zlabel('Volatilité implicite')
    ax2.set_title('Surface modélisée (Heston)')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig("volatility_analysis_plots/1_volatility_surface.png", dpi=150, bbox_inches='tight')
    print("   → Graphique sauvegardé: volatility_analysis_plots/1_volatility_surface.png")
    plt.close()
    
    print("\n2. Génération des smiles de volatilité par maturité...")
    sample_df_grouped = sample_df.copy()
    sample_df_grouped["T_rounded"] = sample_df_grouped["T"].round(4)
    unique_maturities = sorted(sample_df_grouped["T_rounded"].unique())
    n_maturities = len(unique_maturities)
    
    if n_maturities < 4:
        print(f"   → Avertissement: Seulement {n_maturities} maturité(s) unique(s) trouvée(s)")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, T in enumerate(unique_maturities[:4]):
        mask = sample_df_grouped["T_rounded"] == T
        subset = sample_df_grouped[mask].sort_values("moneyness")
        
        if len(subset) == 0:
            axes[idx].text(0.5, 0.5, f'Aucune donnée pour\nmaturité {T*12:.1f} mois',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'Smile de volatilité - Maturité: {T*12:.1f} mois')
            continue
        
        print(f"   → Maturité {T*12:.1f} mois: {len(subset)} options observées")
        
        axes[idx].scatter(subset["moneyness"], subset["iv_observed"], 
                         s=80, marker='o', label='Marché (observé)', 
                         color='blue', alpha=0.7, zorder=3)
        axes[idx].scatter(subset["moneyness"], subset["iv_model"], 
                         s=80, marker='s', label='Modèle Heston (observé)', 
                         color='orange', alpha=0.7, zorder=3)
        
        min_moneyness_obs = subset["moneyness"].min()
        max_moneyness_obs = subset["moneyness"].max()
        moneyness_range = max_moneyness_obs - min_moneyness_obs
        
        min_moneyness_grid = max(0.7, min_moneyness_obs - 0.3 * moneyness_range)
        max_moneyness_grid = min(1.4, max_moneyness_obs + 0.3 * moneyness_range)
        moneyness_grid = np.linspace(min_moneyness_grid, max_moneyness_grid, 50)
        
        print(f"      Simulation pour {len(moneyness_grid)} points de moneyness...")
        iv_model_grid = []
        S0_ref = subset["underlying_price"].iloc[0]
        
        for m in moneyness_grid:
            K = m * S0_ref
            try:
                price = heston_model.european_call_mc(
                    S0=S0_ref, K=K, r=risk_free_rate, T=T,
                    kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
                    n_paths=n_paths, N=n_steps, random_state=42
                )
                iv = implied_vol(price, S0_ref, K, risk_free_rate, T)
                iv_model_grid.append(iv)
            except:
                iv_model_grid.append(np.nan)
        
        iv_model_grid = np.array(iv_model_grid)
        
        valid_mask = ~np.isnan(iv_model_grid)
        moneyness_valid = moneyness_grid[valid_mask]
        iv_model_valid = iv_model_grid[valid_mask]
        
        if len(moneyness_valid) > 0:
            axes[idx].plot(moneyness_valid, iv_model_valid, '-', 
                          label='Modèle Heston (grille élargie)', 
                          linewidth=2.5, color='orange', alpha=0.6, zorder=1)
        
        if len(subset) >= 3:
            try:
                subset_sorted = subset.sort_values("moneyness")
                f_market = interp1d(subset_sorted["moneyness"], subset_sorted["iv_observed"], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
                iv_market_interp = f_market(moneyness_grid)
                valid_interp = ~np.isnan(iv_market_interp) & (iv_market_interp > 0)
                if np.sum(valid_interp) > 0:
                    axes[idx].plot(moneyness_grid[valid_interp], iv_market_interp[valid_interp], 
                                  '--', label='Marché (interpolé)', 
                                  linewidth=2, color='blue', alpha=0.5, zorder=2)
            except:
                pass
        
        axes[idx].set_xlabel('Moneyness (K/S₀)')
        axes[idx].set_ylabel('Volatilité implicite')
        axes[idx].set_title(f'Smile de volatilité - Maturité: {T*12:.1f} mois')
        axes[idx].legend(loc='best', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(min_moneyness_grid * 0.98, max_moneyness_grid * 1.02)
    
    plt.tight_layout()
    plt.savefig("volatility_analysis_plots/2_volatility_smiles.png", dpi=150, bbox_inches='tight')
    print("   → Graphique sauvegardé: volatility_analysis_plots/2_volatility_smiles.png")
    plt.close()
    
    print("\n3. Génération de la courbe de terme de volatilité...")
    at_the_money = sample_df[(sample_df["moneyness"] >= 0.95) & 
                            (sample_df["moneyness"] <= 1.05)].copy()
    
    if len(at_the_money) > 0:
        at_the_money = at_the_money.sort_values("T")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(at_the_money["T"], at_the_money["iv_observed"], 'o-', 
               label='Marché', linewidth=2, markersize=8)
        ax.plot(at_the_money["T"], at_the_money["iv_model"], 's-', 
               label='Modèle Heston', linewidth=2, markersize=8)
        ax.set_xlabel('Maturité (années)')
        ax.set_ylabel('Volatilité implicite')
        ax.set_title('Term Structure de Volatilité (Moneyness ≈ 1.0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("volatility_analysis_plots/3_term_structure.png", dpi=150, bbox_inches='tight')
        print("   → Graphique sauvegardé: volatility_analysis_plots/3_term_structure.png")
        plt.close()
    else:
        print("   → Avertissement: Pas assez d'options à la monnaie pour le term structure")
    
    print("\n4. Génération des graphiques d'erreurs résiduelles...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(sample_df["price_error_pct"], bins=15, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='r', linestyle='--', linewidth=2, label='Erreur nulle')
    axes[0, 0].set_xlabel('Erreur relative de prix (%)')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution des erreurs de prix relatives')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogramme des erreurs de volatilité implicite
    axes[0, 1].hist(sample_df["iv_error"], bins=15, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Erreur nulle')
    axes[0, 1].set_xlabel('Erreur de volatilité implicite')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des erreurs de volatilité implicite')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot des erreurs de prix
    axes[1, 0].boxplot(sample_df["price_error_pct"], vert=True)
    axes[1, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_ylabel('Erreur relative de prix (%)')
    axes[1, 0].set_title('Box plot des erreurs de prix relatives')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Erreur en fonction de la maturité et moneyness
    scatter = axes[1, 1].scatter(sample_df["moneyness"], sample_df["T"], 
                                  c=np.abs(sample_df["iv_error"]), 
                                  cmap='Reds', s=100, alpha=0.7, edgecolors='black')
    axes[1, 1].set_xlabel('Moneyness (K/S₀)')
    axes[1, 1].set_ylabel('Maturité (années)')
    axes[1, 1].set_title('Erreur absolue de volatilité implicite')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig("volatility_analysis_plots/4_residual_errors.png", dpi=150, bbox_inches='tight')
    print("   → Graphique sauvegardé: volatility_analysis_plots/4_residual_errors.png")
    plt.close()
    
    print("\n5. Génération des graphiques de sensibilité des paramètres...")
    ref_T = sample_df["T"].median()
    ref_moneyness = np.linspace(0.85, 1.15, 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].set_title(f'Sensibilité à ρ (corrélation) - T={ref_T:.3f} ans')
    for rho_val in [-0.9, -0.5, -0.1, 0.0]:
        ivs = []
        for m in ref_moneyness:
            K = m * underlying_price
            price = heston_model.european_call_mc(
                S0=underlying_price, K=K, r=risk_free_rate, T=ref_T,
                kappa=kappa, theta=theta, xi=xi, rho=rho_val, v0=v0,
                n_paths=n_paths, N=n_steps, random_state=42
            )
            iv = implied_vol(price, underlying_price, K, risk_free_rate, ref_T)
            ivs.append(iv)
        axes[0, 0].plot(ref_moneyness, ivs, label=f'ρ = {rho_val:.1f}', linewidth=2)
    axes[0, 0].set_xlabel('Moneyness (K/S₀)')
    axes[0, 0].set_ylabel('Volatilité implicite')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title(f'Sensibilité à ξ (volatilité de la volatilité) - T={ref_T:.3f} ans')
    for xi_val in [0.1, 0.3, 0.6, 1.0]:
        ivs = []
        for m in ref_moneyness:
            K = m * underlying_price
            price = heston_model.european_call_mc(
                S0=underlying_price, K=K, r=risk_free_rate, T=ref_T,
                kappa=kappa, theta=theta, xi=xi_val, rho=rho, v0=v0,
                n_paths=n_paths, N=n_steps, random_state=42
            )
            iv = implied_vol(price, underlying_price, K, risk_free_rate, ref_T)
            ivs.append(iv)
        axes[0, 1].plot(ref_moneyness, ivs, label=f'ξ = {xi_val:.1f}', linewidth=2)
    axes[0, 1].set_xlabel('Moneyness (K/S₀)')
    axes[0, 1].set_ylabel('Volatilité implicite')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title(f'Sensibilité à κ (vitesse de retour) - T={ref_T:.3f} ans')
    for kappa_val in [0.5, 2.0, 5.0, 15.0]:
        ivs = []
        for m in ref_moneyness:
            K = m * underlying_price
            price = heston_model.european_call_mc(
                S0=underlying_price, K=K, r=risk_free_rate, T=ref_T,
                kappa=kappa_val, theta=theta, xi=xi, rho=rho, v0=v0,
                n_paths=n_paths, N=n_steps, random_state=42
            )
            iv = implied_vol(price, underlying_price, K, risk_free_rate, ref_T)
            ivs.append(iv)
        axes[1, 0].plot(ref_moneyness, ivs, label=f'κ = {kappa_val:.1f}', linewidth=2)
    axes[1, 0].set_xlabel('Moneyness (K/S₀)')
    axes[1, 0].set_ylabel('Volatilité implicite')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title(f'Sensibilité à θ (variance long terme) - T={ref_T:.3f} ans')
    for theta_val in [0.2, 0.4, 0.6, 0.8]:
        ivs = []
        for m in ref_moneyness:
            K = m * underlying_price
            price = heston_model.european_call_mc(
                S0=underlying_price, K=K, r=risk_free_rate, T=ref_T,
                kappa=kappa, theta=theta_val, xi=xi, rho=rho, v0=v0,
                n_paths=n_paths, N=n_steps, random_state=42
            )
            iv = implied_vol(price, underlying_price, K, risk_free_rate, ref_T)
            ivs.append(iv)
        axes[1, 1].plot(ref_moneyness, ivs, label=f'θ = {theta_val:.1f}', linewidth=2)
    axes[1, 1].set_xlabel('Moneyness (K/S₀)')
    axes[1, 1].set_ylabel('Volatilité implicite')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("volatility_analysis_plots/5_parameter_sensitivity.png", dpi=150, bbox_inches='tight')
    print("   → Graphique sauvegardé: volatility_analysis_plots/5_parameter_sensitivity.png")
    plt.close()
    
    print("\n6. Génération des trajectoires simulées...")
    T_max = sample_df["T"].max()
    n_paths_plot = 5
    
    S_paths, v_paths = heston_model.simulate_heston_paths(
        S0=underlying_price, v0=v0, r=risk_free_rate,
        kappa=kappa, theta=theta, xi=xi, rho=rho,
        T=T_max, N=n_steps, n_paths=n_paths_plot, random_state=123
    )
    
    time_axis = np.linspace(0, T_max, n_steps + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for i in range(n_paths_plot):
        axes[0].plot(time_axis, S_paths[i], alpha=0.7, linewidth=1.5)
    axes[0].axhline(underlying_price, color='r', linestyle='--', linewidth=2, label=f'Prix initial: ${underlying_price:.0f}')
    axes[0].set_xlabel('Temps (années)')
    axes[0].set_ylabel('Prix du sous-jacent S(t)')
    axes[0].set_title(f'Trajectoires simulées du prix (κ={kappa:.2f}, θ={theta:.3f}, ξ={xi:.2f}, ρ={rho:.2f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for i in range(n_paths_plot):
        axes[1].plot(time_axis, v_paths[i], alpha=0.7, linewidth=1.5)
    axes[1].axhline(theta, color='r', linestyle='--', linewidth=2, label=f'Variance long terme: θ={theta:.3f}')
    axes[1].axhline(v0, color='g', linestyle='--', linewidth=2, label=f'Variance initiale: v₀={v0:.3f}')
    axes[1].set_xlabel('Temps (années)')
    axes[1].set_ylabel('Variance v(t)')
    axes[1].set_title('Trajectoires simulées de la variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("volatility_analysis_plots/6_simulated_paths.png", dpi=150, bbox_inches='tight')
    print("   → Graphique sauvegardé: volatility_analysis_plots/6_simulated_paths.png")
    plt.close()
    
    print("\n" + "="*80)
    print("✓ Tous les graphiques ont été générés dans le dossier 'volatility_analysis_plots/'")
    print("="*80)


def main():
    # 1. Chargement et préparation des données
    data = load_and_prepare_data("data.csv")
    print(f"Nombre total d'options chargées : {len(data)}")

    # 2. Filtrage des options pour garder un ensemble homogène
    # Écarter les options trop proches de l'échéance (T < 0.02 ans ≈ 1 semaine)
    # et trop loin de la monnaie (moneyness hors de [0.8, 1.2])
    filtered_data = filter_homogeneous_options(
        data,
        min_maturity=0.02,      # Au moins ~1 semaine de maturité
        min_moneyness=0.8,      # Au moins 80% du spot
        max_moneyness=1.2       # Au plus 120% du spot
    )
    print(f"Nombre d'options après filtrage (T ≥ 0.02 ans, moneyness ∈ [0.8, 1.2]) : {len(filtered_data)}")

    # 3. Sélection d’un sous‑ensemble homogène pour le calibrage
    # Définition des maturités cibles représentatives
    target_maturities = [
        0.083,   # 1 mois (≈0.083 ans)
        0.25,    # 3 mois (≈0.25 ans)
        0.5,     # 6 mois (≈0.5 ans)
        0.75,    # 9 mois (≈0.75 ans)
    ]
    
    print("\nSélection des options autour des maturités cibles :")
    sample_df = select_homogeneous_options(
        filtered_data,
        target_maturities=target_maturities,
        moneyness_tolerance=0.15,      # Moneyness entre 0.85 et 1.15 (±15%)
        strikes_per_maturity=4,        # 4 strikes par maturité
        random_state=42
    )
    
    if sample_df.empty:
        raise RuntimeError("Aucune option sélectionnée pour la calibration. Vérifiez les filtres.")

    print(f"\nNombre total d'options sélectionnées pour la calibration : {len(sample_df)}")

    # 4. Calibration du modèle de Heston
    print("\n" + "="*80)
    print("CALIBRATION DU MODÈLE DE HESTON")
    print("="*80)
    res = calibrate_heston(
        sample_df,
        risk_free_rate=0.02,
        n_paths=5000,          # Augmenté à 5000 trajectoires
        n_steps=100,           # Augmenté à 100 pas de temps
        maxiter=200,           # Augmenté à 200 itérations
        n_initial_points=5,    # Test de 5 points de départ différents
        verbose=True,
    )
    print("\n" + "="*80)
    print("PARAMÈTRES CALIBRÉS")
    print("="*80)
    print(f"  kappa (vitesse de retour à la moyenne) : {res.x[0]:.4f}")
    print(f"    → Borne: [0.5, 15.0] {'[À LA BORNE]' if res.x[0] <= 0.5001 or res.x[0] >= 14.9999 else '[INTERNE]'}")
    print(f"  theta (variance de long terme)         : {res.x[1]:.4f}")
    vol_long_term = np.sqrt(res.x[1]) * 100
    print(f"    → Borne: [0.1, 0.8] {'[À LA BORNE]' if res.x[1] <= 0.1001 or res.x[1] >= 0.7999 else '[INTERNE]'}")
    print(f"    → Volatilité de long terme: {vol_long_term:.2f}%")
    print(f"  xi (volatilité de la volatilité)       : {res.x[2]:.4f}")
    print(f"    → Borne: [0.05, 1.0] {'[À LA BORNE]' if res.x[2] <= 0.0501 or res.x[2] >= 0.9999 else '[INTERNE]'}")
    print(f"  rho (corrélation)                      : {res.x[3]:.4f}")
    print(f"    → Borne: [-0.9, 0.0] {'[À LA BORNE]' if res.x[3] <= -0.8999 or res.x[3] >= -0.0001 else '[INTERNE]'}")
    print(f"  v0 (variance initiale)                 : {res.x[4]:.4f}")
    vol_init = np.sqrt(res.x[4]) * 100
    print(f"    → Borne: [0.01, 1.0] {'[À LA BORNE]' if res.x[4] <= 0.0101 or res.x[4] >= 0.9999 else '[INTERNE]'}")
    print(f"    → Volatilité initiale: {vol_init:.2f}%")

    # 5. Comparaison prix modèle / prix marché et calcul des volatilités implicites
    print("\n" + "="*80)
    print("CALCUL DES PRIX ET VOLATILITÉS IMPLICITES AVEC LES PARAMÈTRES CALIBRÉS")
    print("="*80)
    print("Utilisation de 10000 trajectoires et 100 pas de temps pour une précision maximale...")
    
    sample_df = sample_df.copy()
    model_prices = []
    model_ivs = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        mp = heston_model.european_call_mc(
            S0=row["underlying_price"],
            K=row["strike"],
            r=0.02,
            T=row["T"],
            kappa=res.x[0],
            theta=res.x[1],
            xi=res.x[2],
            rho=res.x[3],
            v0=res.x[4],
            n_paths=10000,      # Augmenté à 10000 pour la précision finale
            N=100,              # Utiliser les mêmes pas de temps que la calibration
            random_state=123,
        )
        model_prices.append(mp)
        # Calcule la volatilité implicite du prix modèle
        iv_m = implied_vol(mp, row["underlying_price"], row["strike"], 0.02, row["T"])
        model_ivs.append(iv_m)
        
        if (i + 1) % 4 == 0:
            print(f"  Traitement de l'option {i + 1}/{len(sample_df)}...")

    sample_df["model_price"] = model_prices
    sample_df["iv_model"] = model_ivs

    # 6. Affichage des résultats comparatifs
    print("\n" + "="*80)
    print("RÉSULTATS COMPARATIFS (PRIX OBSERVÉS VS PRIX MODÈLE)")
    print("="*80)
    display_cols = ["instrument_name", "strike", "T", "moneyness", "price_usd", "model_price"]
    print(sample_df[display_cols].to_string(index=False))
    
    # Calcul des erreurs relatives
    sample_df["price_error_pct"] = 100 * (sample_df["model_price"] - sample_df["price_usd"]) / sample_df["price_usd"]
    print(f"\nErreur moyenne relative : {sample_df['price_error_pct'].abs().mean():.2f}%")
    print(f"Erreur maximale relative : {sample_df['price_error_pct'].abs().max():.2f}%")

    print("\n" + "="*80)
    print("VOLATILITÉS IMPLICITES (OBSERVÉES VS MODÈLE)")
    print("="*80)
    iv_cols = ["instrument_name", "T", "moneyness", "iv_observed", "iv_model"]
    print(sample_df[iv_cols].to_string(index=False))
    
    # Calcul des erreurs de volatilité implicite
    sample_df["iv_error"] = sample_df["iv_model"] - sample_df["iv_observed"]
    print(f"\nErreur moyenne de volatilité implicite : {sample_df['iv_error'].abs().mean():.4f}")
    print(f"Erreur maximale de volatilité implicite : {sample_df['iv_error'].abs().max():.4f}")
    
    # 7. Génération des visualisations d'analyse de la surface de volatilité
    underlying_price_mean = sample_df["underlying_price"].mean()
    plot_volatility_analysis(
        sample_df=sample_df,
        calibrated_params=res.x,
        underlying_price=underlying_price_mean,
        risk_free_rate=0.02,
        n_paths=5000,  # Utiliser moins de trajectoires pour les graphiques de sensibilité (plus rapide)
        n_steps=100
    )


if __name__ == "__main__":
    main()