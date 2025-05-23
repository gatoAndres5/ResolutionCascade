// renderer.js
const { ipcRenderer } = require('electron');

window.addEventListener('DOMContentLoaded', () => {
  const configBtn = document.getElementById('go-to-config');
  if (configBtn) {
    configBtn.addEventListener('click', () => {
      ipcRenderer.send('navigate-to-config');
    });
  }
});
