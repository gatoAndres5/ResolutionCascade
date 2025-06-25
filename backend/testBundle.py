import numpy as np
from scipy.special import jv

def compute_bundle_mtf(Xi, Eta, d_core, d_spacing):
    theta = np.deg2rad(60)
    u = Xi * np.cos(theta) + Eta * np.sin(theta)
    x_samp = d_spacing
    y_samp = np.sqrt(3) * d_spacing
    u_samp = d_spacing

    P = np.sqrt(Xi**2 + Eta**2)

    # π-normalized sinc in NumPy
    sampling = np.abs(
        np.sinc(Xi * x_samp) *
        np.sinc(Eta * y_samp) *
        np.sinc(u * u_samp)
    )

    argument = np.pi * d_core * P
    with np.errstate(divide='ignore', invalid='ignore'):
        fiber = np.abs(2 * jv(1, argument) / argument)
        fiber[np.isnan(fiber)] = 1

    MTF1 = sampling * fiber
    return np.nan_to_num(MTF1, nan=1.0)


def run_mtf_test():
    # Inputs (assumed in mm)
    d_core = 0.002     # 10 microns
    d_spacing = 0.003 # 12 microns

    freq_vals = np.linspace(-250, 250, 501)  # Covers full frequency range
    Xi, Eta = np.meshgrid(freq_vals, freq_vals)
    MTF1 = compute_bundle_mtf(Xi, Eta, d_core, d_spacing)

    # Get center row (ξ direction) and column (η direction)
    center_idx = len(freq_vals) // 2
    xi_slice = MTF1[center_idx, center_idx:]  # row → ξ direction
    eta_slice = MTF1[center_idx:, center_idx]  # col → η direction
    freqs = freq_vals[center_idx:]  # 0 to 250

    # Query values
    target_freqs = [50, 100, 150, 200, 250]
    print("MTF Values (Bundle Model):")
    for f in target_freqs:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  Frequency {f:>3} cycles/mm → ξ (E): {xi_slice[idx]:.4f}, η (N): {eta_slice[idx]:.4f}")


if __name__ == "__main__":
    run_mtf_test()
