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
    # Load matrix (for shape only — we just care about dimensions)
    with open("matrix_output.json", "r") as f:
        matrix = json.load(f)

    height = len(matrix)
    width = len(matrix[0])

    # Load config to get binning factor
    with open("resolution_config.json", "r") as f:
        config = json.load(f)

    # Get binning and pixel size from the first detector element
    binning = 1.0
    pixSizeX = 1.0
    pixSizeY = 1.0

    for element in config:
        if element.get("type") == "Detector":
            bin_str = element.get("binning", "1")
            try:
                binning = eval(bin_str)  # Safe only if you control input
            except Exception:
                print("Failed to parse binning. Defaulting to 1.")
                binning = 1.0

            try:
                pixSizeX = float(element.get("pixel_x", "1"))
                pixSizeY = float(element.get("pixel_y", "1"))
            except ValueError:
                print("Failed to parse pixel sizes. Defaulting to 1.0.")
                pixSizeX = pixSizeY = 1.0
            break

    # Constants using parsed values
    wx4 = pixSizeX * binning
    wy4 = pixSizeY * binning

    # Generate coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    Xi, Eta = np.meshgrid(x, y)

    # Calculate MTF3
    mtf3 = np.abs(np.sinc(Xi * wx4)) * np.abs(np.sinc(Eta * wy4))

    # Optionally save MTF3 to file
    with open("build/mtf3_output.json", "w") as f:
        json.dump(mtf3.tolist(), f)

    print("MTF3 calculation complete. Saved to mtf3_output.json.")

def lensMTF():
    # Load matrix to determine dimensions
    with open("matrix_output.json", "r") as f:
        matrix = json.load(f)

    height = len(matrix)
    width = len(matrix[0])

    # Load config to get wavelength and NA
    with open("resolution_config.json", "r") as f:
        config = json.load(f)

    wavelength = 1.0
    NA = 1.0

    for element in config:
        if element.get("type") == "Lens":
            try:
                wavelength = float(element.get("wavelength", "1"))
                NA = float(element.get("na", "1"))
            except ValueError:
                print("Failed to parse wavelength or NA. Defaulting to 1.0.")
                wavelength = NA = 1.0
            break

    # Generate coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    Xi, Eta = np.meshgrid(x, y)

    # Compute radial distance R
    R = np.sqrt(Xi**2 + Eta**2)

    # Compute phi with domain-clamped argument
    argument = np.clip((wavelength * R) / (2 * NA), -1.0, 1.0)
    with np.errstate(invalid='ignore'):
        phi = np.arccos(argument)

    # MTF2 formula
    mtf2 = (2 * (phi - np.cos(phi) * np.sin(phi))) / np.pi
    mtf2 = np.nan_to_num(mtf2)

    # Save result to file
    with open("build/mtf2_output.json", "w") as f:
        json.dump(mtf2.tolist(), f)

    print("MTF2 calculation complete. Saved to mtf2_output.json.")

def bundleMTF():
    # Load matrix to determine dimensions
    with open("build/matrix_output.json", "r") as f:
        matrix = json.load(f)

    height = len(matrix)
    width = len(matrix[0])

    # Load bundle parameters from config
    with open("build/resolution_config.json", "r") as f:
        config = json.load(f)

    core_diameter = 1.0
    spacing = 1.0

    for element in config:
        if element.get("type") == "Bundle":
            try:
                core_diameter = float(element.get("core_diameter", "1"))
                spacing = float(element.get("core_spacing", "1"))
            except ValueError:
                print("Invalid core parameters. Using default values.")
            break

    r_core = core_diameter / 2

    # Create coordinate grid centered at zero
    x = np.linspace(-width // 2, width // 2, width)
    y = np.linspace(-height // 2, height // 2, height)
    Xi, Eta = np.meshgrid(x, y)
    rho = np.sqrt(Xi**2 + Eta**2)

    # Avoid division by zero in Bessel argument
    with np.errstate(divide='ignore', invalid='ignore'):
        argument = np.pi * r_core * rho
        bessel = np.ones_like(rho)
        mask = argument != 0
        bessel[mask] = (2 * j1(argument[mask])) / argument[mask]

    mtf1 = (np.pi * r_core**2) * bessel
    mtf1 = np.nan_to_num(mtf1)

    # Save output
    with open("build/mtf1_output.json", "w") as f:
        json.dump(mtf1.tolist(), f)

    print("MTF1 (bundle) calculation complete. Saved to mtf1_output.json.")

if __name__ == "__main__":
    create_matrix()
    bundleMTF()

