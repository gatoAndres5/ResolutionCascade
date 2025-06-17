import json
import math

import numpy as np
from scipy.special import j1


def create_matrix():
    matrix = [[0 for _ in range(1000)] for _ in range(1000)]
    # Save to JSON file
    with open("build/matrix_output.json", "w") as f:
        json.dump(matrix, f)

    print("Matrix saved to matrix_output.json")

def doSomething():
    result = []
    for row_index in range(1000):
        row = []
        for col_index in range(1000):
            row.append(col_index * 5)  # Multiply x-coordinate (column) by 5
        result.append(row)

    with open("matrix_modified.json", "w") as f:
        json.dump(result, f)
    print("Modified matrix saved to matrix_modified.json")


def detectorMTF():
    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)

    height = len(matrix)
    width = len(matrix[0])

    # Load config to get binning, pixel size, and magnification
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    pixel_size = 0.00454  # mm (default)
    magnification = 1.0
    binning = 1.0

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
    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)
    height = len(matrix)
    width = len(matrix[0])

    # Load lens parameters from config
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    wavelength = 0.000550  # default: mm
    NA = 0.25              # default
    magnification = 1.0    # default

    for element in config:
        if element.get("type") == "Lens":
            try:
                wavelength = float(element.get("wavelength", wavelength))
                NA = float(element.get("na", NA))
                magnification = float(element.get("magnification", magnification))
            except ValueError:
                print("Invalid lens config. Using defaults.")
            break

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
    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)
    height = len(matrix)
    width = len(matrix[0])

    # Load bundle parameters from config
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    # Default values
    d_core = 0.002  # mm
    d_spacing = 0.003  # mm
    d_img = 0.327  # mm

    for element in config:
        if element.get("type") == "Bundle":
            try:
                d_core = float(element.get("core_diameter", d_core))
                d_spacing = float(element.get("core_spacing", d_spacing))
                d_img = float(element.get("image_circle", d_img))
            except ValueError:
                print("Invalid bundle parameters. Using defaults.")
            break

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

if __name__ == "__main__":
    create_matrix()
    detectorMTF()

