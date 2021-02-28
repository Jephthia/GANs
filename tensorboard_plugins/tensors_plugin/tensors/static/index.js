import './vue.js'
import './app.js';

export async function render() {
  const stylesheet = document.createElement('link');
  stylesheet.rel = 'stylesheet';
  stylesheet.href = './static/style.css';
  document.body.appendChild(stylesheet);

  const root = document.createElement('div')
  root.id = 'app'
  root.innerHTML = `<app></app>`
  document.body.appendChild(root);

  const app = new Vue({
    el: '#app',
  })
}
