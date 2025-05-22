const { app, BrowserWindow } = require('electron');
const path = require('path');





function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      nodeIntegration: true
    }
  });

  // Load your GUI from local file or from Django
  win.loadFile('index.html');

}

app.whenReady().then(() => {
  createWindow();
});
