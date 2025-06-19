
#  Resolution Cascade


---

##  Requirements

###  Python
- **Version:** Python 3.7 or higher
- **Package Manager:** `pip` (comes with Python)
- **Virtual Environments:** `venv` (recommended)

 **Install Python:**
- [Windows/macOS/Linux Installer](https://www.python.org/downloads/)
-  Make sure to check **"Add Python to PATH"** during installation on Windows.

---

###  Node.js
- **Version:** Node.js v16+ recommended
- **Package Manager:** `npm` (included with Node.js)
- **Electron:** Install manually via npm

 **Install Node.js:**
- [Node.js Download Page](https://nodejs.org/)
- Use the **LTS version** for stability.

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gatoAndres5/ResolutionCascade.git
cd ResolutionCascade
````

---

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

---

### 3. Set Up Electron

```bash
# Install Node.js dependencies
npm install

# (If you get 'electron' not found, install globally)
npm install -g electron
```

---

### 4. Run the App

```bash
npm start
```

---

## Usage Flow

1. Enter the number of elements (Bundle, Lens, Detector)
2. Define parameters for each component
3. Click **Generate**
4. View and navigate MTF graphs with:

   * ← / → to switch between MTF1, MTF2, MTF3
   * 3D/2D toggle to change view mode

---

## 🗂 Project Structure

```
ResolutionCascade/
├── backend/                    # Python MTF calculation scripts
│   └── matrices.py
├── build/                      # Output folder for mtf1/2/3 JSONs
├── config.html                 # Main Electron frontend interface
├── renderer.js                 # JS logic for DOM/Plotly
├── resolution_config.json      # Saved user configuration
├── requirements.txt            # Python dependencies
├── package.json                # NPM config for Electron
└── main.js                     # Electron entrypoint
```

---

## 📘 Python Dependencies

Listed in `requirements.txt`:

```
numpy
scipy
plotly
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🛠 Troubleshooting

* **Electron not recognized**: Run `npm install -g electron`
* **Python module missing**: Run `pip install -r requirements.txt` in the virtual environment
* **Graphs not updating**: Ensure you click **Generate** after setting parameters
* **Matrix doesn't load**: Ensure MTF files exist in `/build` directory before viewing

---

##  Tips

* All JSON configuration files are located under `build/`
* Only the **positive quadrant** of MTF is visualized for clarity
* Default plots use `matrix_output.json` as the dimension reference

---




