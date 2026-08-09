import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import sys

from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip, mad_std
from scipy.stats import linregress
from scipy.optimize import curve_fit
from ckmeans_1d_dp import ckmeans

import warnings
warnings.filterwarnings("ignore")

# Validating User's Input
def get_valid_input(prompt_text, expected_type, valid_options=None):
    while True:
        try:
            user_input = input(prompt_text).strip()
            
            if expected_type == str and valid_options:
                user_input = user_input.lower()
                if user_input not in valid_options:
                    raise ValueError(f"Please enter one of the following: {'/'.join(valid_options)}")
            elif expected_type == int:
                user_input = int(user_input)
            elif expected_type == float:
                user_input = float(user_input)
            
            return user_input
            
        except ValueError as e:
            if str(e).startswith("Please enter"):
                print(f"-> Error: {e}")
            else:
                data_type = "an integer" if expected_type == int else "a float (decimal number)"
                print(f"-> Error: Invalid input. Please enter {data_type}.")
        except KeyboardInterrupt:
            # Menangani jika pengguna menekan Ctrl+C saat input
            print("\n\n[Program force-stopped by user (Ctrl+C)]")
            sys.exit(0)


# Optional Step: Removing Distance Effect from Magnitude
def get_reducted_magnitude(df_lc: pd.DataFrame, df_eph: pd.DataFrame) -> pd.DataFrame:
    df_out = df_lc.copy()

    app_mag = df_lc['mag'].values
    r = np.interp(df_lc['jd'], df_eph['jd'], df_eph['helio_dist'])
    delta = np.interp(df_lc['jd'], df_eph['jd'], df_eph['geo_dist'])

    reducted_mag = app_mag - (5 * np.log10(r * delta))

    df_out['mag'] = reducted_mag

    return df_out

# Step 1: Interpolating Phase Angle from Ephemeris Data
def interpolating(df_lc: pd.DataFrame, df_eph: pd.DataFrame) -> pd.DataFrame:
    df_out = df_lc.copy()

    alpha = np.interp(df_lc['jd'], df_eph['jd'], df_eph['phase_angle'])

    df_out['phase_angle'] = alpha

    return df_out

# Step 2: Clustering Data Based on Phase Angle
def clustering(df_lc: pd.DataFrame, manual: str, num_cluster: int = 0) -> any:
    df_small = df_lc[df_lc['phase_angle'] <= 7]
    df_large = df_lc[df_lc['phase_angle'] > 7]

    # For small angle, use uniform binning with range 1 degree
    cluster_edges = np.arange(0, 7, 1)
    df_small['cluster'] = pd.cut(df_small['phase_angle'], bins=cluster_edges)
    df_small = df_small.dropna(subset=['cluster']).copy()
    df_small['cluster'] = df_small['cluster'].astype(str)

    print("Clustering small phase angle... Success!")

    # For large phase angles, use Ckmeans algorithm (Wang & Song, 2011)
    phase_large = df_large['phase_angle'].astype(np.float64).values
    if manual == 'n':
        min_data = 50
        max_k = len(phase_large) // min_data # Make sure each cluster have at least 50 data
        upper_bound = min(11, max(3, max_k + 1))

        model = ckmeans(phase_large, k=(2, upper_bound))
        df_large['cluster'] = model.cluster
    elif manual == 'y':
        model = ckmeans(phase_large, k=num_cluster)
        df_large['cluster'] = model.cluster

    print("Clustering large phase angle... Success!")

    # Renaming cluster column into phase angle range
    mean_alphas = df_large.groupby('cluster')['phase_angle'].mean()
    sorted_cluster = mean_alphas.sort_values().index
    mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(sorted_cluster)}
    df_large['cluster'] = df_large['cluster'].map(mapping)

    interval_mapping = {}
    for b in sorted(df_large['cluster'].unique()):
        min_alpha = int(np.floor(df_large[df_large['cluster'] == b]['phase_angle'].min()))
        max_alpha = int(np.ceil(df_large[df_large['cluster'] == b]['phase_angle'].max()))
        interval_mapping[b] = f"({min_alpha}, {max_alpha}]"
    
    df_large['cluster'] = df_large['cluster'].map(interval_mapping)

    # Comclustere both dataframe into one
    df_out = pd.concat([df_small, df_large], ignore_index=True).sort_values(by='jd').reset_index(drop=True)

    return df_out, model

# Plotting clustering result from clustered data
def plot_clustering(df_cluster):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_cluster,
        x='phase_angle',
        y='mag', 
        hue='cluster',
        palette='tab20',
        s=60,
        alpha=1
    )
    plt.gca().invert_yaxis()

    plt.xlim(left=0)
    plt.xlabel('Phase Angle (deg)', fontsize=12)
    plt.ylabel('Reduced Magnitude', fontsize=12)
    plt.title('Clustering Result', fontsize=14)
    plt.legend(title='Cluster Range', bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Step 3: Finding Best Rotation Period using Periodogram Lomb-Scargle
def periodogram(df_lc: pd.DataFrame) -> any:
    # Centering magnitude to eliminate distance and phase angle effect
    mag_center = df_lc.groupby('cluster')['mag'].transform(lambda x: x - x.mean())

    mag = mag_center
    jd = df_lc['jd'].values
    err = df_lc['err'].values

    # Start Lomb-Scargle Periodogram
    ls = LombScargle(jd, mag, err)
    min_freq = 1 / (100 / 24.0)
    max_freq = 1 / (2.0 / 24.0)
    freq, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq)

    best_idx = np.argmax(power)
    best_freq = freq[best_idx]

    best_period_days = (1 / best_freq)
    best_period_hours = best_period_days * 24

    use_period = 2 * best_period_days   # Times 2 because most asteroids have two peak

    print(f"Best rotation period: {best_period_hours * 2:.4f} hours ({best_period_days * 2:.5f} days)")
    
    return best_period_days, use_period, freq, power

def plot_periodogram(freq: np.array , power: np.array, best_period_days: float):
    periods = (1 / freq) * 24
    best_period_hours = best_period_days * 24

    plt.figure(figsize=(10, 5))
    plt.plot(periods, power, color='blue', lw=1.5)
    plt.axvline(best_period_hours, color='red', linestyle='--', label=f'Best: {best_period_hours:.2f} h')
    plt.xlabel('Period (hours)', fontsize=12)
    plt.ylabel('Lomb-Scargle Power', fontsize=12)
    plt.title('Lomb-Scargle Periodogram', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Step 4: Finding Best Fourier Order Using Reduced Chi-Squared as Parameter
def fourier_matrix(t: np.array, period: float, order: int, t0=None) -> any:
    N = len(t)
    if t0 is None:
        t0 = np.median(t)
    X = np.ones((N, 1))

    for i in range(1, order + 1):
        cos_term = np.cos(2 * np.pi * i * (t - t0) / period)
        sin_term = np.sin(2 * np.pi * i * (t - t0) / period)
        X = np.column_stack((X, cos_term, sin_term))
    return X

def get_best_order(df_lc: pd.DataFrame, period: float) -> any:
    jd = df_lc['jd'].values
    mag = df_lc['mag'].values
    err = df_lc['err'].values

    chi2_list = []
    orders = range(1, 6)
    for order in orders:
        X = fourier_matrix(jd, period, order)

        W = 1 / err
        X_w = X * W[:, None]
        mag_w = mag * W

        coeffs, *_ = np.linalg.lstsq(X_w, mag_w, rcond=None)
        mag_fit = X @ coeffs

        dof = len(jd) - (2 * order + 1)
        chi2 = np.sum(((mag - mag_fit) / err)**2)
        chi2_reduced = chi2 / dof
        chi2_list.append(chi2_reduced)

    best_order = orders[1]
    best_drop = 0
    for i in range(1, len(orders)):
        drop = chi2_list[i - 1] - chi2_list[i]
        if drop > best_drop:
            best_drop = drop
            best_order = orders[i]
    
    print(f"Best fourier order: {best_order}")

    return best_order, chi2_list, orders

def plot_chi2(chi2s: np.array, orders: np.array):
    plt.figure(figsize=(8, 6))
    plt.plot(orders, chi2s, c='blue')
    plt.scatter(orders, chi2s, c='black', s=50)
    plt.xticks(np.arange(1, 6, 1))
    plt.xlabel('Fourier Orders')
    plt.ylabel('Reduced Chi-squared')
    plt.title('Finding Best Fourier Orders')
    plt.grid(alpha=0.8)
    plt.show()

# Step 5: Fitting Lightcurve for Each Clusters Using Fourier Series
def extract_c0(df_lc: pd.DataFrame, period: float, order: int) -> any:
    c0_data = []
    plot_data = []

    for cluster in df_lc['cluster'].unique():
        df_cluster = df_lc[df_lc['cluster'] == cluster].copy()

        num_params = 2 * order + 1
        if len(df_cluster) <= num_params:
            continue

        jd = df_cluster['jd']
        mag = df_cluster['mag']
        err = df_cluster['err']
        mean_alpha = np.mean(df_cluster['phase_angle'])
        t0 = np.median(jd)

        # Initial fitting
        X = fourier_matrix(jd, period, order, t0=t0)
        W = np.array(1.0 / err)
        X_w = X * W[:, np.newaxis]
        mag_w = mag * W
        
        coeffs, *_ = np.linalg.lstsq(X_w, mag_w, rcond=None)

        # Sigma clipping to remove outliers
        mag_fit_initial = X @ coeffs
        residuals = mag - mag_fit_initial

        filtered_data = sigma_clip(residuals, sigma=3, maxiters=3, stdfunc=mad_std)
        valid_mask = ~filtered_data.mask

        num_outliers = len(mag) - valid_mask.sum()
        if num_outliers > 0:
            print(f'Removing {num_outliers} outliers from cluster {cluster}')

        # Final fitting
        jd_clean = jd[valid_mask]
        mag_clean = mag[valid_mask]
        err_clean = err[valid_mask]
        t0_clean = np.median(jd_clean)
        
        if len(jd_clean) <= num_params:
            continue
            
        phase_obs = ((jd_clean - t0_clean) / period) % 1.0
        
        phase_sorted = np.sort(phase_obs)
        
        gaps = np.append(np.diff(phase_sorted), 1.0 - phase_sorted[-1] + phase_sorted[0])
        max_gap = np.max(gaps)

        if max_gap > 0.5:
            print(f"Skipping cluster {cluster} with max gap: {max_gap:.2f}")
            continue

        X_clean = fourier_matrix(jd_clean, period, order, t0=t0_clean)
        W_clean = np.array(1.0 / err_clean)
        X_w_clean = X_clean * W_clean[:, np.newaxis]
        mag_w_clean = mag_clean * W_clean

        coeffs_final, *_ = np.linalg.lstsq(X_w_clean, mag_w_clean, rcond=None)

        jd_check = np.linspace(t0_clean, period + t0_clean, 200)
        X_check = fourier_matrix(jd_check, period, order, t0=t0_clean)
        mag_check = X_check @ coeffs_final

        if (np.max(mag_check) - np.min(mag_check)) > 1:
            continue

        # Extracting c0 and their error
        c0 = coeffs_final[0]
        cov_matrix = np.linalg.inv(X_w_clean.T @ X_w_clean)
        c0_err = np.sqrt(cov_matrix[0, 0])

        plot_data.append({
            'cluster': cluster,
            'phase_angle': mean_alpha,
            'jd': jd_clean,
            'mag': mag_clean,
            'coeffs': coeffs_final,
            't0': t0_clean,
            'c0': c0
        })

        c0_data.append({
            'clusters': cluster,
            'phase_angle': mean_alpha,
            'mag': c0,
            'err': c0_err,
            'n_obs': len(jd_clean)
        })
    
    if not c0_data:
        print("Warning: No clusters met the criteria for Fourier Series fitting (c0_data is empty).")
        return pd.DataFrame(), []
    
    df_out = pd.DataFrame(c0_data)
    df_out = df_out.sort_values('phase_angle').reset_index(drop=True)
    df_out = df_out[df_out['err'] < 0.2]

    return df_out, plot_data

def plot_fourier_fit(plot: list, period: float, order: int):
    num_plots = len(plot)
    if num_plots == 0:
        print("There are no data point to plot.")
        return
    
    cols = 4
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))

    axes = np.atleast_1d(axes).flatten()
    
    line_fourier = None
    line_c0 = None

    for i, data in enumerate(plot):
        ax = axes[i]

        t = data['jd']
        mag = data['mag']
        coeffs = data['coeffs']
        t0 = data['t0']
        c0 = data['c0']
        phase_obs = ((t - t0) / period) % 1

        phase_fit = np.linspace(0, 1, 200)
        t_fit = t0 + phase_fit * period
        X_fit = fourier_matrix(t_fit, period, order, t0=t0)
        mag_fit = X_fit @ coeffs

        ax.scatter(phase_obs, mag, color='gray', s=15, alpha=0.7)
        line_fourier, = ax.plot(phase_fit, mag_fit, color='blue', lw=2) 
        line_c0 = ax.axhline(c0, color='red', linestyle='--', lw=1.5)

        ax.invert_yaxis()
        ax.grid(True, linestyle=':', alpha=0.6)

        info_text = f"{data['cluster']} | $\\alpha$ = {data['phase_angle']:.1f}$^\\circ$"
        ax.text(0.5, 0.95, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Rotation Phase", fontsize=14, y=0.03)
    fig.supylabel("Reducted Magnitude", fontsize=14, x=0.02)

    if line_fourier and line_c0:
        fig.legend(
            [line_fourier, line_c0], 
            ['Fourier Fit', '$a_0$ Baseline'], 
            loc='lower center', 
            ncol=2, 
            fontsize=12,
            frameon=True, 
            bbox_to_anchor=(0.8, 0.01)
        )

    plt.tight_layout(rect=[0.03, 0.05, 1, 1], h_pad=1.5, w_pad=1.5)
    plt.show()

# Step 6: Calculate H-G1-G2
def phi1(phase_angle: np.array) -> np.array:
    phase_rad = np.radians(phase_angle)
    return 1 - (6 * phase_rad / np.pi)

def phi2(phase_angle: np.array) -> np.array:
    phase_rad = np.radians(phase_angle)
    return 1 - (9 * phase_rad / (5 * np.pi))

def phi3(phase_angle: np.array) -> np.array:
    phase_rad = np.radians(phase_angle)
    return np.exp(-4 * np.pi * (np.tan(phase_rad / 2))**(2 / 3))

def HG1G2_model(phase_angle: np.array, H: float, G1: float, G2: float) -> np.array:
    term = G1 * phi1(phase_angle) + G2 * phi2(phase_angle) + (1 - G1 - G2) * phi3(phase_angle)
    term = np.clip(term, 10e-10, None)
    return H - 2.5 * np.log10(term)

def find_parameter(df_lc: pd.DataFrame) -> any:
    phase = df_lc['phase_angle'].values
    mag = df_lc['mag'].values
    err = df_lc['err'].values

    # H-G1-G2 model
    popt, pcov = curve_fit(
        HG1G2_model,
        xdata=phase,
        ydata=mag,
        sigma=err,
        absolute_sigma=True,
        p0=[min(mag) - 0.5, 0.2, 0.2],
        bounds=([-np.infty, 0, 0], [np.infty, 1, 1]),
        maxfev=10000
    )
    H, G1, G2 = popt
    H_err, G1_err, G2_err = np.sqrt(np.diag(pcov))

    # Linear model
    linear_model = linregress(phase, mag)
    slope = linear_model.slope
    H_lin = linear_model.intercept
    H_lin_err = linear_model.stderr

    print(f"Absolute Magnitude (H) = {H:.3f} ± {H_err:.3f} mag")
    print(f'Absolute Magnitude with Linear Model (H) = {H_lin:.3f} ± {H_lin_err:.3f} mag')
    print(f"Slope Parameter (G1)  = {G1:.3f} ± {G1_err:.3f}")
    print(f"Slope Parameter (G2)  = {G2:.3f} ± {G2_err:.3f}")

    params = [H, G1, G2, H_lin, slope]
    return params

def plot_phase_curve(df_lc: pd.DataFrame, params: list):
    phase = df_lc['phase_angle'].values
    mag = df_lc['mag'].values
    err = df_lc['err'].values

    H, G1, G2, H_lin, slope = params

    phase_fit = np.linspace(0, max(phase) + 1, 200)
    mag_fit = HG1G2_model(phase_fit, H, G1, G2)
    mag_lin = slope * phase_fit + H_lin

    plt.figure(figsize=(8, 8))
    plt.errorbar(
        phase, mag, yerr=err, 
        fmt='o', color='black', markersize=6, capsize=3, alpha=0.8, 
        label='Extracted $c_0$ Data'
    )
    plt.plot(phase_fit, mag_fit, c='blue', label=f'H-G1-G2 Model with H = {H:.3f}, G1 = {G1:.3f}, G2 = {G2:.3f}')
    plt.plot(phase_fit, mag_lin, c='red', linestyle='--', label='Linear Model')
    plt.grid(True, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel('Phase Angle (deg)', fontsize=15)
    plt.ylabel('Reducted Magnitude', fontsize=15)
    plt.title(f'Phase Curve', fontsize=17)
    plt.legend()
    plt.show()