const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

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
});
