const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

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
    const filePath = path.join(__dirname, 'resolution_config.json');
    fs.writeFileSync(filePath, JSON.stringify(configData, null, 2));
    console.log("Configuration saved to:", filePath);
  });
});
