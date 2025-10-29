const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile('index.html');

  process.on('uncaughtException', (err) => {
    console.error(err);
    dialog.showErrorBox('Uncaught Exception', String(err));
  });
  process.on('unhandledRejection', (err) => {
    console.error(err);
    dialog.showErrorBox('Unhandled Rejection', String(err));
  });
}

app.whenReady().then(() => {
  createWindow();

  ipcMain.on('navigate-to-config', () => {
    mainWindow.loadFile(path.join(__dirname, 'config.html'));
  });

  ipcMain.on('navigate-to-home', () => {
    mainWindow.loadFile('index.html');
  });

  ipcMain.on('save-config', (event, configData) => {
    //  Writable path
    const buildDir = path.join(__dirname, 'build');
    const filePath = path.join(buildDir, 'resolution_config.json');

    if (!fs.existsSync(buildDir)) {
      fs.mkdirSync(buildDir, { recursive: true });
    }
    fs.writeFileSync(filePath, JSON.stringify(configData, null, 2));
    console.log('Configuration saved to:', filePath);

    //  Resolve backend path correctly in dev vs. packaged
    // We need the py file to be OUTSIDE the asar at runtime
    const resourcesBase = app.isPackaged ? process.resourcesPath : __dirname;
    const pyScript = path.join(resourcesBase, 'backend', 'matrices.py');

    //  Try to spawn Python safely; show a helpful message if missing
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const child = spawn(pythonCmd, [pyScript, filePath], { stdio: 'pipe' });

    child.stdout.on('data', (d) => console.log(`[Python] ${d}`));
    child.stderr.on('data', (d) => console.error(`[Python Error] ${d}`));

    child.on('error', (err) => {
      console.error('Failed to start Python:', err);
      dialog.showErrorBox(
        'Python not found',
        'Could not start Python. Make sure Python is installed and on PATH, or bundle an embedded Python.'
      );
    });

    child.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      if (code !== 0) {
        dialog.showErrorBox('Python exited with error', `Exit code: ${code}`);
      }
    });
  });
});
