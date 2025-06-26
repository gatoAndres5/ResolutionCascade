import json
import math
import os

import numpy as np
from scipy.special import jv


def create_matrix():
    matrix = [[0 for _ in range(1000)] for _ in range(1000)]
    print("Current working directory:", os.getcwd())
    # Make build dir
    os.makedirs("build", exist_ok=True)
    # Save to JSON file
    with open("build/matrix_output.json", "w") as f:
        json.dump(matrix, f)

    print("Matrix saved to matrix_output.json")

def detectorMTF():
    output_path = "build/mtf4_output.json"

    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)

    height = len(matrix)
    width = len(matrix[0])

    # Load config for pixel size and magnification
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    pixel_width_um = None
    pixel_pitch_um = None
    magnification = None

    for element in config:
        if element.get("type") == "Detector":
            try:
                pixel_width_um = float(element.get("pixel_size", pixel_width_um))  
                pixel_pitch_um = float(element.get("pixel_pitch", pixel_pitch_um))
            except ValueError:
                print("Invalid pixel size in detector config.")
        if element.get("type") in ["Lens (Focusing Optics)", "Lens (Relay Optics)"]:
            try:
                magnification = float(element.get("magnification", magnification))
            except ValueError:
                print("Invalid magnification value.")

    if pixel_width_um is None or magnification is None or pixel_pitch_um is None:
        ones_matrix = [[1.0 for _ in range(width)] for _ in range(height)]
        with open(output_path, "w") as f:
            json.dump(ones_matrix, f)
        print("MTF4 skipped: missing config values. Output is a matrix of all 1s.")
        return

    # Compute scaled pixel and pitch in mm
    wx = (pixel_width_um / 1000) / magnification
    pitch = wx if pixel_pitch_um is None else (pixel_pitch_um / 1000) / magnification

    # Build frequency grid
    x = np.linspace(-((width - 1) / 2), ((width - 1) / 2), width)
    y = np.linspace(-((height - 1) / 2), ((height - 1) / 2), height)
    # Build frequency grid (cycles/mm)
    freq_vals = np.linspace(-250, 250, width)
    Xi, Eta = np.meshgrid(freq_vals, freq_vals)

    # Compute MTF
    footprint = np.abs(np.sinc(Xi * wx)) * np.abs(np.sinc(Eta * wx))
    sample = np.abs(np.sinc(Xi * pitch)) * np.abs(np.sinc(Eta * pitch))
    mtf3 = np.abs(footprint * sample)

    # Normalize to DC
    max_val = mtf3[height // 2][width // 2]
    if max_val != 0:
        mtf3 /= max_val

    # Save result
    with open(output_path, "w") as f:
        json.dump(mtf3.tolist(), f)

    print("MTF4 (detector) calculation complete. Saved to mtf4_output.json.")

    # Print MTF values at target spatial frequencies
    center_idx = width // 2
    mtf_slice = mtf3[height // 2, center_idx:]
    freqs = freq_vals[center_idx:]

    print("MTF Values (Detector Model):")
    for f in [50, 100, 150, 200, 250]:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  Frequency {f:>3} cycles/mm → MTF: {mtf_slice[idx]:.4f}")
        
def lensRelayMTF(): 
    output_path = "build/mtf2_output.json"

    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)
    height = len(matrix)
    width = len(matrix[0])

    # Load lens parameters from config
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    wavelength = None
    NA = None              
    magnification = None  

    for element in config:
        if element.get("type") == "Lens (Relay Optics)":
            try:
                wavelength = float(element.get("wavelength", wavelength))
                NA = float(element.get("na", NA))
                magnification = float(element.get("magnification", magnification))
            except ValueError:
                print("Invalid lens config. Using defaults.")
            break
    
    # If any required value is missing, output 1s and exit
    if wavelength is None or NA is None or magnification is None:
        ones_matrix = [[1.0 for _ in range(width)] for _ in range(height)]
        with open(output_path, "w") as f:
            json.dump(ones_matrix, f)
        print("MTF2 skipped: missing config values. Output is a matrix of all 1s.")
        return

    # Build frequency coordinate grid (in cycles/mm)
    freq_vals = np.linspace(-250, 250, width)
    Xi, Eta = np.meshgrid(freq_vals, freq_vals)

    # Radial frequency
    P = np.sqrt(Xi**2 + Eta**2)

    # Compute phi and MTF
    argument = np.clip((wavelength * P) / (2 * NA), -1.0, 1.0)
    with np.errstate(invalid='ignore'):
        phi = np.arccos(argument)

    MTF2 = 2 * (phi - np.cos(phi) * np.sin(phi)) / np.pi
    MTF2 = np.real(np.abs(MTF2))
    MTF2 = np.nan_to_num(MTF2)

    # Normalize to center value
    center_val = MTF2[height // 2][width // 2]
    if center_val != 0:
        MTF2 /= center_val

    # Save matrix
    with open(output_path, "w") as f:
        json.dump(MTF2.tolist(), f)

    print("MTF2 (lens) calculation complete. Saved to mtf2_output.json.")

    # Extract and print 1D MTF along ξ direction
    center_idx = width // 2
    mtf_slice = MTF2[center_idx, center_idx:]
    freqs = freq_vals[center_idx:]

    target_freqs = [50, 100, 150, 200, 250]
    print("MTF Values (Relay Optics Model):")
    for f in target_freqs:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  Frequency {f:>3} cycles/mm → MTF: {mtf_slice[idx]:.4f}")

def lensFocusingMTF():
    output_path = "build/mtf3_output.json"

    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)
    height = len(matrix)
    width = len(matrix[0])

    # Load lens parameters from config
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    wavelength = None
    NA = None              
    magnification = None  

    for element in config:
        if element.get("type") == "Lens (Focusing Optics)":
            try:
                wavelength = float(element.get("wavelength", wavelength))
                NA = float(element.get("na", NA))
                magnification = float(element.get("magnification", magnification))
            except ValueError:
                print("Invalid lens config. Using defaults.")
            break
    
    # If any required value is missing, output 1s and exit
    if wavelength is None or NA is None or magnification is None:
        ones_matrix = [[1.0 for _ in range(width)] for _ in range(height)]
        with open(output_path, "w") as f:
            json.dump(ones_matrix, f)
        print("MTF3 skipped: missing config values. Output is a matrix of all 1s.")
        return

    # Build frequency coordinate grid (in cycles/mm)
    freq_vals = np.linspace(-250, 250, width)
    Xi, Eta = np.meshgrid(freq_vals, freq_vals)

    # Radial frequency
    P = np.sqrt(Xi**2 + Eta**2)

    # Compute phi and MTF
    argument = np.clip((wavelength * P) / (2 * NA), -1.0, 1.0)
    with np.errstate(invalid='ignore'):
        phi = np.arccos(argument)

    MTF3 = 2 * (phi - np.cos(phi) * np.sin(phi)) / np.pi
    MTF3 = np.real(np.abs(MTF3))
    MTF3 = np.nan_to_num(MTF3)

    # Normalize to center value
    center_val = MTF3[height // 2][width // 2]
    if center_val != 0:
        MTF3 /= center_val

    # Save matrix
    with open(output_path, "w") as f:
        json.dump(MTF3.tolist(), f)

    print("MTF3 (focusing lens) calculation complete. Saved to mtf3_output.json.")

    # Extract and print 1D MTF along ξ direction
    center_idx = width // 2
    mtf_slice = MTF3[center_idx, center_idx:]
    freqs = freq_vals[center_idx:]

    target_freqs = [50, 100, 150, 200, 250]
    print("MTF Values (Focusing Lens Model):")
    for f in target_freqs:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  Frequency {f:>3} cycles/mm → MTF: {mtf_slice[idx]:.4f}")


def bundleMTF():
    output_path = "build/mtf1_output.json"

    # Load matrix to determine shape
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)
    height = len(matrix)
    width = len(matrix[0])

    # Load config
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    d_core = d_spacing = None
    for item in config:
        if item.get("type") == "Bundle":
            try:
                d_core = float(item["core_diameter"])
                d_spacing = float(item["core_spacing"])
            except (ValueError, KeyError):
                print("Invalid bundle parameters.")
            break

    if d_core is None or d_spacing is None:
        ones = [[1.0 for _ in range(width)] for _ in range(height)]
        with open(output_path, "w") as f:
            json.dump(ones, f)
        print("⚠️ Missing bundle values. Output is all 1s.")
        return

    # Frequency grid [-250, 250] to match your working test
    freq_vals = np.linspace(-250, 250, width)
    Xi, Eta = np.meshgrid(freq_vals, freq_vals)

    # --- MTF computation (EXACTLY as in your test) ---
    theta = np.deg2rad(60)
    u = Xi * np.cos(theta) + Eta * np.sin(theta)
    x_samp = d_spacing
    y_samp = np.sqrt(3) * d_spacing
    u_samp = d_spacing

    P = np.sqrt(Xi**2 + Eta**2)

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
    MTF1 = np.nan_to_num(MTF1, nan=1.0)

    # Normalize to DC
    dc_value = MTF1[height // 2, width // 2]
    if dc_value > 0:
        MTF1 /= dc_value

    # Save output
    with open(output_path, "w") as f:
        json.dump(MTF1.tolist(), f)

    print("✅ MTF1 (bundle) written to", output_path)

    # === Print diagnostic values (exact same as test) ===
    center_idx = width // 2
    xi_slice = MTF1[center_idx, center_idx:]   # ξ direction (row)
    eta_slice = MTF1[center_idx:, center_idx]  # η direction (col)
    freqs = freq_vals[center_idx:]             # Positive freqs only

    target_freqs = [50, 100, 150, 200, 250]
    print("📊 MTF Values (Bundle):")
    for f in target_freqs:
        idx = np.argmin(np.abs(freqs - f))
        print(f"  Frequency {f:>3} cycles/mm → ξ (E): {xi_slice[idx]:.4f}, η (N): {eta_slice[idx]:.4f}")

def combinedMTF():
    files = {
        "mtf1": "build/mtf1_output.json",
        "mtf2": "build/mtf2_output.json",
        "mtf3": "build/mtf3_output.json",
        "mtf4": "build/mtf4_output.json"
    }

    # Check that all files exist
    for key, path in files.items():
        if not os.path.exists(path):
            print(f"{key} file not found: {path}")
            return

    # Load and convert each matrix to a NumPy array
    mtf1 = np.array(json.load(open(files["mtf1"], "r")))
    mtf2 = np.array(json.load(open(files["mtf2"], "r")))
    mtf3 = np.array(json.load(open(files["mtf3"], "r")))
    mtf4 = np.array(json.load(open(files["mtf4"], "r")))

    # Ensure all matrices are the same shape
    if not (mtf1.shape == mtf2.shape == mtf3.shape == mtf4.shape):
        print("Error: MTF matrices are not the same shape.")
        return

    # Element-wise multiplication
    mtf_combined = mtf1 * mtf2 * mtf3 * mtf4

    # Save result
    with open("build/mtf5_output.json", "w") as f:
        json.dump(mtf_combined.tolist(), f)

    print("Combined MTF saved to build/mtf5_output.json")

if __name__ == "__main__":
    create_matrix()
    bundleMTF()
    lensRelayMTF()
    lensFocusingMTF()
    detectorMTF()
    combinedMTF()

