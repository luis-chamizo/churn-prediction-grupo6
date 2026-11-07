# =====================================================
# ESTILO DE LA INTERFAZ (CSS + detector de tema claro/oscuro)
# Módulo liviano, sin dependencias pesadas: Inicio.py lo carga antes que
# cualquier página para que el estilo esté presente desde el primer instante.
# =====================================================
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent


# Detecta el tema activo (claro/oscuro) por la luminancia del fondo de la app y lo
# expone como <html data-st-theme="...">, para que estilo.css se adapte al instante
# cuando el usuario cambia el tema en el menú ⋮ (sin recargar la página).
THEME_WATCHER = """
<script>
(function () {
  if (window.__stThemeWatch) { window.__stThemeWatch(); return; }
  function lum(c) {
    var m = (c || '').match(/\\d+(\\.\\d+)?/g);
    if (!m || m.length < 3 || (m.length > 3 && +m[3] === 0)) return null;
    return (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
  }
  function check() {
    var app = document.querySelector('[data-testid="stApp"]') || document.body;
    var l = lum(getComputedStyle(app).backgroundColor);
    if (l === null) l = lum(getComputedStyle(document.body).backgroundColor);
    if (l === null) return;
    var t = l < 0.5 ? 'dark' : 'light';
    if (document.documentElement.getAttribute('data-st-theme') !== t)
      document.documentElement.setAttribute('data-st-theme', t);
  }
  // Streamlit guarda la elección System/Light/Dark por URL (clave stActiveTheme-RUTA-v2).
  // (Nota: no escribir el signo menor-que seguido de una letra en este script: el
  //  sanitizador de st.html elimina el bloque completo.)
  // Cuando el usuario la cambia, se replica a todas las páginas de la app para que
  // se mantenga en toda la aplicación y en futuras visitas desde este navegador.
  try {
    var origSet = Storage.prototype.setItem;
    Storage.prototype.setItem = function (k, v) {
      origSet.call(this, k, v);
      var m = /^stActiveTheme-(.*)-(v\\d+)$/.exec(k);
      if (this !== window.localStorage || !m) return;
      var rutas = { '/': 1 };
      document.querySelectorAll('[data-testid="stSidebarNav"] a[href]').forEach(function (a) {
        try { rutas[new URL(a.href).pathname] = 1; } catch (e) {}
      });
      for (var r in rutas) {
        var kk = 'stActiveTheme-' + r + '-' + m[2];
        if (kk !== k) origSet.call(this, kk, v);
      }
    };
  } catch (e) {}
  var pend = false;
  function sched() { if (!pend) { pend = true; requestAnimationFrame(function () { pend = false; check(); }); } }
  window.__stThemeWatch = sched;
  new MutationObserver(sched).observe(document.head, { childList: true, subtree: true, characterData: true });
  setInterval(check, 1000);
  check();
})();
</script>
"""


def apply_style():
    css = (APP_DIR / "estilo.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.html(THEME_WATCHER, unsafe_allow_javascript=True, width="content")
