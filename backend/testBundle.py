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

def compute_lens_mtf(Xi, Eta, wavelength, NA):
    P = np.sqrt(Xi**2 + Eta**2)
    argument = np.clip((wavelength * P) / (2 * NA), -1.0, 1.0)

    with np.errstate(invalid='ignore'):
        phi = np.arccos(argument)

    MTF2 = 2 * (phi - np.cos(phi) * np.sin(phi)) / np.pi
    MTF2 = np.real(np.abs(MTF2))
    return np.nan_to_num(MTF2)


def run_relay_mtf_test():
    wavelength = 0.00055   # 550 nm → mm
    NA = 0.4

    freq_vals = np.linspace(-250, 250, 501)  # cycles/mm
    Xi, Eta = np.meshgrid(freq_vals, freq_vals)
    MTF2 = compute_lens_mtf(Xi, Eta, wavelength, NA)

    center_idx = len(freq_vals) // 2
    mtf_slice = MTF2[center_idx, center_idx:]  # take horizontal center row
    freqs = freq_vals[center_idx:]

    target_freqs = [50, 100, 150, 200, 250]
    print("MTF Values (Relay Optics Model):")
    for f in target_freqs:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  Frequency {f:>3} cycles/mm → MTF: {mtf_slice[idx]:.4f}")

def compute_detector_mtf(Xi, Eta, pixel_width_um, M2, pixel_pitch_um):
    """
    Compute MTF for a detector model, optionally allowing different pixel pitch from pixel width.

    Parameters:
    - Xi, Eta: spatial frequency meshgrids (cycles/mm)
    - pixel_width_um: pixel width (µm)
    - M2: magnification factor
    - pixel_pitch_um: optional separate pixel pitch (µm)

    Returns:
    - Normalized MTF3 matrix
    """
    wx = (pixel_width_um / 1000) / M2  # convert µm → mm, scale
    pitch = wx if pixel_pitch_um is None else (pixel_pitch_um / 1000) / M2

    footprint = np.abs(np.sinc(Xi * wx)) * np.abs(np.sinc(Eta * wx))
    sample = np.abs(np.sinc(Xi * pitch)) * np.abs(np.sinc(Eta * pitch))
    MTF3 = np.abs(footprint * sample)

    # Normalize to 1
    max_val = np.max(MTF3)
    if max_val != 0:
        MTF3 /= max_val
    return MTF3


def run_detector_mtf_test():
    pixel_width_um = 3.45  # User-provided pixel width in microns
    pixel_pitch_um = 3.45  # Set to a number like 4.0 if different from pixel width
    M2 = 8.5  # magnification

    freq_vals = np.linspace(-250, 250, 501)  # cycles/mm
    Xi, Eta = np.meshgrid(freq_vals, freq_vals)
    MTF3 = compute_detector_mtf(Xi, Eta, pixel_width_um, M2, pixel_pitch_um)

    center_idx = len(freq_vals) // 2
    mtf_slice = MTF3[center_idx, center_idx:]  # ξ direction
    freqs = freq_vals[center_idx:]

    print("MTF Values (Detector Model):")
    for f in [50, 100, 150, 200, 250]:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  Frequency {f:>3} cycles/mm → MTF: {mtf_slice[idx]:.4f}")


if __name__ == "__main__":
    run_mtf_test()
    run_relay_mtf_test()
    run_detector_mtf_test()
