// renderer.js
const { ipcRenderer } = require('electron');

window.addEventListener('DOMContentLoaded', () => {
  const configBtn = document.getElementById('go-to-config');
  const homeBtn = document.getElementById('go-to-home')
  if (configBtn) {
    configBtn.addEventListener('click', () => {
      ipcRenderer.send('navigate-to-config');
    });
  }
  if (homeBtn) {
    homeBtn.addEventListener('click', () => {
      ipcRenderer.send('navigate-to-home');
    });
  }
});
