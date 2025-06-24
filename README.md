
#  Resolution Cascade

##  Requirements

###  Python
- **Version:** Python 3.7 or higher  
- **Package Manager:** `pip` (included with Python)  
- **Virtual Environments:** `venv` (recommended for project isolation)  

** How to Install Python:**
- Go to [python.org/downloads](https://www.python.org/downloads/)
- Download the installer for your operating system (Windows/macOS/Linux)
- ** Important (Windows only):** During installation, make sure to check the box that says **“Add Python to PATH”**

** Verify Installation:**

```bash
python --version
pip --version
````

---

###  Node.js + Electron

* **Version:** Node.js v16 or newer (LTS recommended)
* **Package Manager:** `npm` (comes with Node.js)
* **Electron:** Installed manually via `npm`

** How to Install Node.js:**

* Visit [nodejs.org](https://nodejs.org/)
* Download and install the **LTS version**
* Restart your terminal afterward

** Verify Installation:**

```bash
node --version
npm --version
```

---

##  Installation & Setup

> These instructions assume you are using a terminal (Command Prompt, PowerShell, or Bash).

###  1. Clone the Project Repository

>  **New to Git?** Download it from [git-scm.com](https://git-scm.com/) and choose default settings during install.

```bash
git clone https://github.com/gatoAndres5/ResolutionCascade.git
cd ResolutionCascade
```

---

### 2. Set Up Python Environment

```bash
# Create a virtual environment named 'venv'
python -m venv venv
```

**Activate it:**

```bash
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

You should see `(venv)` appear in your terminal. Now install required packages:

```bash
pip install -r requirements.txt
```

---

### 3. Set Up Electron

```bash
# Install Node.js dependencies from package.json
npm install
```

>  If you see `'electron' is not recognized as a command`, run:

```bash
npm install -g electron
```

---

### 4. Run the App

```bash
npm start
```

This should launch the Electron desktop app.

---

## Terminal Tips 

* **`cd folder-name`** → change into a folder
* **`ls`** or **`dir`** → list files in a folder
* **`mkdir folder-name`** → make a new folder
* If your terminal gets stuck, press **Ctrl+C** to cancel a command
* To rerun the app later, make sure to re-activate the virtual environment

---

##  Having Issues?

* Make sure Python and Node are installed properly (`--version` commands)
* Restart your terminal after installing new software
* Always activate your `venv` before running Python commands

---



## Usage Flow

1. Enter the number of elements (Bundle, Lens, Detector)
2. Define parameters for each component
3. Click **Generate**
4. View and navigate MTF graphs with:

   * ← / → to switch between MTF1, MTF2, MTF3
   * 3D/2D toggle to change view mode

---

##  Project Structure

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

##  Python Dependencies

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

##  Troubleshooting

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




