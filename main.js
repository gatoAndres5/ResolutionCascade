const { app, BrowserWindow, ipcMain } = require('electron');
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
      contextIsolation: false,  // Needed for ipcRenderer in this setup
    },
  });

  mainWindow.loadFile('index.html');
}

app.whenReady().then(() => {
  createWindow();

  ipcMain.on('navigate-to-config', () => {
    mainWindow.loadFile('config.html');
  });

  ipcMain.on('navigate-to-home', () => {
    mainWindow.loadFile('index.html');
  });

  ipcMain.on('save-config', (event, configData) => {
    const buildDir = path.join(__dirname, 'build');
    const filePath = path.join(buildDir, 'resolution_config.json');

    //  Ensure build directory exists
    if (!fs.existsSync(buildDir)) {
      fs.mkdirSync(buildDir, { recursive: true });
    }

    //  Save config file
    fs.writeFileSync(filePath, JSON.stringify(configData, null, 2));
    console.log("Configuration saved to:", filePath);

    //  Run Python process
    const pythonProcess = spawn('python', [path.join(__dirname, 'backend/matrices.py'), filePath]);

    pythonProcess.stdout.on('data', (data) => {
      console.log(`[Python] ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`[Python Error] ${data}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
    });
});
  
});
