/* ---------------------------------------------------------------------------
 * web/main.js
 *
 * Role in BackendRace: the entire browser-side driver.
 *   * Declares window.Module BEFORE backendrace.js loads so Emscripten picks up
 *     our onRuntimeInitialized hook.
 *   * Converts mouse/touch pixel coordinates into grid coordinates and calls
 *     into WASM through Module.ccall (brBrush / brAddCell).
 *   * Binds keys 1-4 to the exported C++ brSetBackend() so the compute backend
 *     can be swapped mid-simulation.
 *   * Runs a requestAnimationFrame loop that calls the exported brStep() and
 *     blits the RGBA frame straight out of WASM linear memory (Module.HEAPU8)
 *     with putImageData, then scales it onto the visible canvas.
 *   * Maintains the live FPS / step-time / cells-per-second overlay.
 *
 * Assumes backendrace.js + backendrace.wasm sit next to this file.
 * --------------------------------------------------------------------------- */

'use strict';

/* Emscripten configuration object. MUST exist before backendrace.js runs. */
var Module = {
  print:    (t) => console.log('[wasm]', t),
  printErr: (t) => console.warn('[wasm]', t),
  onRuntimeInitialized: () => boot(),
};
window.Module = Module;

(function () {
  // ----------------------------- constants --------------------------------
  const GRID_W = 512;          // multiples of 32 (tile size)
  const GRID_H = 512;
  const BACKEND_COUNT = 4;

  // ------------------------------- state ----------------------------------
  const S = {
    ready: false,
    paused: false,
    material: 1,               // 1 = Sand
    brush: 6,
    backend: 0,
    pointerDown: false,
    lastGX: -1,
    lastGY: -1,
    fps: 0,
    lastFrameTs: 0,
    throughput: 0,
    pixelPtr: 0,
    imageData: null,
    offCanvas: null,
    offCtx: null,
    api: {},
  };

  // ------------------------------ DOM refs --------------------------------
  const canvas   = document.getElementById('sim');
  const ctx      = canvas.getContext('2d', { alpha: false });
  const bootEl   = document.getElementById('boot');
  const ovBack   = document.getElementById('ov-backend');
  const ovFps    = document.getElementById('ov-fps');
  const ovStep   = document.getElementById('ov-step');
  const ovThru   = document.getElementById('ov-throughput');
  const ovGrid   = document.getElementById('ov-grid');
  const ovNote   = document.getElementById('ov-note');
  const brushEl  = document.getElementById('brush');
  const brushVal = document.getElementById('brush-val');

  // ------------------------------ boot ------------------------------------
  function boot() {
    // cwrap every exported C function once; ccall is used for the hot ones too.
    S.api = {
      init:        Module.cwrap('brInit',             null,     ['number', 'number']),
      step:        Module.cwrap('brStep',             null,     []),
      render:      Module.cwrap('brRender',           null,     []),
      brush:       Module.cwrap('brBrush',            null,     ['number', 'number', 'number', 'number']),
      addCell:     Module.cwrap('brAddCell',          null,     ['number', 'number', 'number']),
      setBackend:  Module.cwrap('brSetBackend',       'number', ['number']),
      backendName: Module.cwrap('brBackendName',      'string', ['number']),
      backendAvail:Module.cwrap('brBackendAvailable', 'number', ['number']),
      backendDesc: Module.cwrap('brBackendDesc',      'string', ['number']),
      pixels:      Module.cwrap('brPixels',           'number', []),
      width:       Module.cwrap('brWidth',            'number', []),
      height:      Module.cwrap('brHeight',           'number', []),
      lastStepMs:  Module.cwrap('brLastStepMs',       'number', []),
      clear:       Module.cwrap('brClear',            null,     []),
      seed:        Module.cwrap('brSeed',             null,     ['number']),
    };

    S.api.init(GRID_W, GRID_H);
    S.api.seed(0xC0FFEE);

    const w = S.api.width();
    const h = S.api.height();

    // Offscreen buffer at native grid resolution; the visible canvas is scaled.
    S.offCanvas = document.createElement('canvas');
    S.offCanvas.width = w;
    S.offCanvas.height = h;
    S.offCtx = S.offCanvas.getContext('2d', { alpha: false });
    S.imageData = S.offCtx.createImageData(w, h);

    ctx.imageSmoothingEnabled = false;
    ovGrid.textContent = `${w}x${h} (${(w / 32) | 0}x${(h / 32) | 0} tiles)`;

    refreshBackendButtons();
    selectBackend(0, true);
    selectMaterial(1);

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    S.ready = true;
    bootEl.classList.add('hidden');
    S.lastFrameTs = performance.now();
    requestAnimationFrame(frame);
  }

  // --------------------------- canvas sizing ------------------------------
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      ctx.imageSmoothingEnabled = false;
    }
  }

  // ------------------------------ frame loop ------------------------------
  function frame(ts) {
    requestAnimationFrame(frame);
    if (!S.ready) return;

    // ---- advance the simulation inside WASM -----------------------------
    if (S.paused) {
      Module.ccall('brRender', null, [], []);
    } else {
      Module.ccall('brStep', null, [], []);
    }

    // ---- blit the RGBA frame out of WASM linear memory ------------------
    // NOTE: with ALLOW_MEMORY_GROWTH the heap can be reallocated, which
    // detaches any cached typed array. Re-read Module.HEAPU8 and the pointer
    // every frame -- it is just a property read, not a copy.
    const ptr = Module.ccall('brPixels', 'number', [], []);
    const w = S.offCanvas.width;
    const h = S.offCanvas.height;
    const bytes = Module.HEAPU8.subarray(ptr, ptr + w * h * 4);
    S.imageData.data.set(bytes);
    S.offCtx.putImageData(S.imageData, 0, 0);
    ctx.drawImage(S.offCanvas, 0, 0, canvas.width, canvas.height);

    // ---- stats ----------------------------------------------------------
    const dt = ts - S.lastFrameTs;
    S.lastFrameTs = ts;
    if (dt > 0) S.fps = S.fps * 0.9 + (1000 / dt) * 0.1;

    const stepMs = S.api.lastStepMs();
    const cells = w * h;
    const inst = stepMs > 0 ? (cells / (stepMs / 1000)) : 0;
    S.throughput = S.throughput * 0.85 + inst * 0.15;

    ovFps.textContent  = S.fps.toFixed(1);
    ovStep.textContent = `${stepMs.toFixed(2)} ms`;
    ovThru.textContent = `${(S.throughput / 1e6).toFixed(1)} Mcell/s`;
  }

  // --------------------------- backend switching --------------------------
  function refreshBackendButtons() {
    document.querySelectorAll('[data-backend]').forEach((btn) => {
      const id = parseInt(btn.dataset.backend, 10);
      const ok = S.api.backendAvail(id) !== 0;
      btn.disabled = !ok;
      btn.title = S.api.backendDesc(id);
    });
  }

  function selectBackend(id, silent) {
    if (id < 0 || id >= BACKEND_COUNT) return;
    const ok = S.api.setBackend(id) !== 0;
    if (!ok) {
      ovNote.textContent = `${S.api.backendName(id)} unavailable in this build`;
      setTimeout(() => { ovNote.textContent = ''; }, 2500);
      return;
    }
    S.backend = id;
    ovBack.textContent = S.api.backendName(id);
    if (!silent) {
      ovNote.textContent = S.api.backendDesc(id);
      setTimeout(() => { ovNote.textContent = ''; }, 3000);
    }
    document.querySelectorAll('[data-backend]').forEach((b) => {
      b.classList.toggle('active', parseInt(b.dataset.backend, 10) === id);
    });
    S.throughput = 0;   // reset the EMA so the new backend is visible instantly
  }

  function selectMaterial(m) {
    S.material = m;
    document.querySelectorAll('[data-material]').forEach((b) => {
      b.classList.toggle('active', parseInt(b.dataset.material, 10) === m);
    });
  }

  // ------------------------------- input ----------------------------------
  function toGrid(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const gx = Math.floor(((clientX - rect.left) / rect.width) * S.offCanvas.width);
    const gy = Math.floor(((clientY - rect.top) / rect.height) * S.offCanvas.height);
    return { gx, gy };
  }

  function paintAt(clientX, clientY) {
    if (!S.ready) return;
    const { gx, gy } = toGrid(clientX, clientY);
    if (gx < 0 || gy < 0 || gx >= S.offCanvas.width || gy >= S.offCanvas.height) return;

    // Interpolate between samples so fast drags leave a continuous stroke.
    if (S.lastGX >= 0) {
      const dx = gx - S.lastGX;
      const dy = gy - S.lastGY;
      const steps = Math.max(Math.abs(dx), Math.abs(dy));
      for (let i = 1; i < steps; ++i) {
        const ix = S.lastGX + Math.round((dx * i) / steps);
        const iy = S.lastGY + Math.round((dy * i) / steps);
        Module.ccall('brBrush', null,
                     ['number', 'number', 'number', 'number'],
                     [ix, iy, S.brush, S.material]);
      }
    }
    Module.ccall('brBrush', null,
                 ['number', 'number', 'number', 'number'],
                 [gx, gy, S.brush, S.material]);
    S.lastGX = gx;
    S.lastGY = gy;
  }

  canvas.addEventListener('mousedown', (e) => {
    S.pointerDown = true;
    S.lastGX = -1;
    paintAt(e.clientX, e.clientY);
    e.preventDefault();
  });
  window.addEventListener('mousemove', (e) => {
    if (S.pointerDown) paintAt(e.clientX, e.clientY);
  });
  window.addEventListener('mouseup', () => {
    S.pointerDown = false;
    S.lastGX = -1;
  });
  canvas.addEventListener('mouseleave', () => { S.lastGX = -1; });

  canvas.addEventListener('touchstart', (e) => {
    S.pointerDown = true;
    S.lastGX = -1;
    const t = e.changedTouches[0];
    paintAt(t.clientX, t.clientY);
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchmove', (e) => {
    const t = e.changedTouches[0];
    paintAt(t.clientX, t.clientY);
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchend', () => {
    S.pointerDown = false;
    S.lastGX = -1;
  });

  window.addEventListener('keydown', (e) => {
    if (!S.ready) return;
    switch (e.key) {
      case '1': case '2': case '3': case '4':
        selectBackend(parseInt(e.key, 10) - 1, false); break;
      case 'q': case 'Q': selectMaterial(1); break;   // Sand
      case 'w': case 'W': selectMaterial(2); break;   // Water
      case 'e': case 'E': selectMaterial(3); break;   // Fire
      case 'r': case 'R': selectMaterial(4); break;   // Wall
      case 'x': case 'X': selectMaterial(0); break;   // Erase
      case 'c': case 'C': S.api.clear(); break;
      case 's': case 'S': S.api.seed((Math.random() * 0xffffffff) >>> 0); break;
      case ' ':
        S.paused = !S.paused;
        ovNote.textContent = S.paused ? 'paused' : '';
        e.preventDefault();
        break;
      default: return;
    }
  });

  document.getElementById('backend-buttons').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-backend]');
    if (btn && !btn.disabled) selectBackend(parseInt(btn.dataset.backend, 10), false);
  });
  document.getElementById('material-buttons').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-material]');
    if (btn) selectMaterial(parseInt(btn.dataset.material, 10));
  });
  brushEl.addEventListener('input', () => {
    S.brush = parseInt(brushEl.value, 10);
    brushVal.textContent = String(S.brush);
  });
})();