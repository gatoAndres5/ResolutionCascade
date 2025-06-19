import json
import math
import os

import numpy as np
from scipy.special import j1


def create_matrix():
    matrix = [[0 for _ in range(1000)] for _ in range(1000)]
    # Save to JSON file
    with open("build/matrix_output.json", "w") as f:
        json.dump(matrix, f)

    print("Matrix saved to matrix_output.json")


def detectorMTF():

    output_path = "build/mtf3_output.json"
    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)

    height = len(matrix)
    width = len(matrix[0])

    # Load config to get binning, pixel size, and magnification
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    pixel_size = None  # mm (default)
    magnification = None
    binning = None

    for element in config:
        if element.get("type") == "Detector":
            try:
                pixel_size = float(element.get("pixel_x", pixel_size))
            except ValueError:
                print("Invalid pixel_x, using default.")

            try:
                binning = eval(element.get("binning", "1"))
            except Exception:
                print("Invalid binning, using 1.")

        if element.get("type") == "Lens":
            try:
                magnification = float(element.get("magnification", magnification))
            except ValueError:
                print("Invalid magnification, using 1.")
    
    # If any required value is missing, skip calculation and copy matrix
    if pixel_size is None or binning is None or magnification is None:
        with open(output_path, "w") as f:
            json.dump(matrix, f)
        print("MTF3 skipped: missing config values. matrix_output.json copied as-is.")
        return

    # Apply binning and magnification
    wx4 = (pixel_size * binning) / magnification
    wy4 = (pixel_size * binning) / magnification

    # Create centered coordinate grid
    x = np.linspace(-((width - 1) / 2), ((width - 1) / 2), width)
    y = np.linspace(-((height - 1) / 2), ((height - 1) / 2), height)
    Xi, Eta = np.meshgrid(x, y)

    # Compute MTF3 using NumPy's sinc (already π-scaled)
    mtf3 = np.abs(np.sinc(Xi * wx4)) * np.abs(np.sinc(Eta * wy4))
    mtf3 = np.nan_to_num(mtf3)

    # Normalize to center (optional)
    center_val = mtf3[height // 2][width // 2]
    if center_val != 0:
        mtf3 /= center_val

    # Save result
    with open("build/mtf3_output.json", "w") as f:
        json.dump(mtf3.tolist(), f)

    print("MTF3 (detector) calculation complete. Saved to mtf3_output.json.")

def lensMTF():
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
        if element.get("type") == "Lens":
            try:
                wavelength = float(element.get("wavelength", wavelength))
                NA = float(element.get("na", NA))
                magnification = float(element.get("magnification", magnification))
            except ValueError:
                print("Invalid lens config. Using defaults.")
            break
    
    # If any required value is missing, skip calculation and copy matrix
    if wavelength is None or NA is None or magnification is None:
        with open(output_path, "w") as f:
            json.dump(matrix, f)
        print("MTF2 skipped: missing config values. matrix_output.json copied as-is.")
        return

    # Generate centered coordinate grid
    x = np.linspace(-((width - 1) / 2), ((width - 1) / 2), width)
    y = np.linspace(-((height - 1) / 2), ((height - 1) / 2), height)
    Xi, Eta = np.meshgrid(x, y)

    # Radial distance from center
    P = np.sqrt(Xi**2 + Eta**2)

    # Compute phi with clipped domain to avoid invalid arccos
    argument = np.clip((wavelength * P) / (2 * NA), -1.0, 1.0)
    with np.errstate(invalid='ignore'):
        phi = np.arccos(argument)

    # Compute MTF2
    MTF2 = (2 * (phi - np.cos(phi) * np.sin(phi))) / np.pi
    MTF2 = np.real(MTF2)
    MTF2 = np.nan_to_num(MTF2)

    # Normalize to max at (0, 0) equivalent: top-left if centered
    max_value = MTF2[height // 2][width // 2]
    if max_value != 0:
        MTF2 /= max_value

    # Save result
    with open("build/mtf2_output.json", "w") as f:
        json.dump(MTF2.tolist(), f)

    print("MTF2 (lens) calculation complete. Saved to mtf2_output.json.")

def bundleMTF():
    output_path = "build/mtf1_output.json"
    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)
    height = len(matrix)
    width = len(matrix[0])

    # Load bundle parameters from config
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    # Default values
    d_core = None  # mm
    d_spacing = None  # mm

    for element in config:
        if element.get("type") == "Bundle":
            try:
                d_core = float(element.get("core_diameter", d_core))
                d_spacing = float(element.get("core_spacing", d_spacing))
            except ValueError:
                print("Invalid bundle parameters. Using defaults.")
            break
    
    # If any required value is missing, skip calculation and copy matrix
    if d_core is None or d_spacing is None:
        with open(output_path, "w") as f:
            json.dump(matrix, f)
        print("MTF1 skipped: missing config values. matrix_output.json copied as-is.")
        return

    r_core = d_core / 2

    # Coordinate grid centered at 0
    x = np.linspace(-((width - 1) / 2), ((width - 1) / 2), width)
    y = np.linspace(-((height - 1) / 2), ((height - 1) / 2), height)

    Xi, Eta = np.meshgrid(x, y)
    P = np.sqrt(Xi**2 + Eta**2)

    # Avoid divide-by-zero at center (0,0)
    argument = np.pi * r_core * P
    denominator = argument
    with np.errstate(divide='ignore', invalid='ignore'):
        bessel_term = j1(argument)
        MTF1 = (np.pi * r_core**2) * (bessel_term / denominator)

    # Normalize and fix NaN at center
    MTF1 = np.nan_to_num(MTF1, nan=1.0)
    MTF1 = np.abs(MTF1)
    max_val = np.max(MTF1)
    if max_val != 0:
        MTF1 /= max_val

    # Save output
    with open("build/mtf1_output.json", "w") as f:
        json.dump(MTF1.tolist(), f)

    print("MTF1 (bundle) calculation complete. Saved to mtf1_output.json.")

def combinedMTF():
    files = {
        "mtf1": "build/mtf1_output.json",
        "mtf2": "build/mtf2_output.json",
        "mtf3": "build/mtf3_output.json"
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

    # Ensure all matrices are the same shape
    if not (mtf1.shape == mtf2.shape == mtf3.shape):
        print("Error: MTF matrices are not the same shape.")
        return

    # Element-wise multiplication
    mtf_combined = mtf1 * mtf2 * mtf3

    # Save result
    with open("build/mtf4_output.json", "w") as f:
        json.dump(mtf_combined.tolist(), f)

    print("Combined MTF saved to build/mtf4_output.json")

if __name__ == "__main__":
    create_matrix()
    bundleMTF()
    lensMTF()
    detectorMTF()
    combinedMTF()

