/* ═══════════════════════════════════════════════════
   EndoScan AI — app.js
   ═══════════════════════════════════════════════════ */

const API_BASE = (() => {
  // Works whether served by FastAPI (same host:8000) or opened directly
  const o = window.location.origin;
  return o.includes(':8000') ? o : o.replace(/:\d+$/, ':8000');
})();

// ── State ──────────────────────────────────────────
const state = {
  modality: 'ct',
  model: 'segmentation',
  files: [],
  result: null,          // last API response
  rawUrl: null,          // /download/… for raw slice PNG
  maskUrl: null,         // /download/… for mask PNG
  maskDownloadUrl: null, // /download/… for .nii.gz
  currentView: 'raw',
};

// ── DOM refs ───────────────────────────────────────
const $  = id => document.getElementById(id);
const fileInput   = $('file-input');
const fileList    = $('file-list');
const fileHint    = $('file-hint');
const dropzone    = $('dropzone');
const runBtn      = $('run-btn');
const logEl       = $('log');
const errMsg      = $('err-msg');
const imgLbl      = $('img-lbl');
const imgStage    = $('img-stage');
const placeholder = $('placeholder');
const phRing      = $('ph-ring');
const viewControls = $('view-controls');
const statsStrip  = $('stats-strip');
const dlRow       = $('dl-row');
const dlMask      = $('dl-mask');
const dlPreview   = $('dl-preview');
const detContent  = $('det-content');
const clsContent  = $('cls-content');
const detCount    = $('det-count');
const clsBadge    = $('cls-badge');
const overlayCanvas = $('overlay-canvas');

// ── Logging ────────────────────────────────────────
function log(msg, cls = '') {
  const s = document.createElement('span');
  if (cls) s.className = cls;
  s.textContent = `> ${msg}\n`;
  logEl.appendChild(s);
  logEl.scrollTop = logEl.scrollHeight;
}

// ── Error display ──────────────────────────────────
function showErr(msg) {
  errMsg.textContent = `⚠ ${msg}`;
  errMsg.style.display = 'block';
}
function clearErr() { errMsg.style.display = 'none'; }

// ── Tabs ───────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    $(`panel-${tab.dataset.tab}`).classList.add('active');
  });
});

// ── Toggle buttons ─────────────────────────────────
document.querySelectorAll('.tog').forEach(btn => {
  btn.addEventListener('click', () => {
    const g = btn.dataset.group;
    document.querySelectorAll(`.tog[data-group="${g}"]`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (g === 'modality') {
      state.modality = btn.dataset.value;
      const max = state.modality === 'ctpet' ? 2 : 1;
      fileHint.textContent = `max ${max}`;
      // Re-validate existing selection
      if (state.files.length) handleFiles(state.files);
    } else {
      state.model = btn.dataset.value;
    }
  });
});

// ── File handling ──────────────────────────────────
// Input is hidden (display:none). Chrome blocks opacity:0 overlays from
// opening file dialogs — trigger programmatically on dropzone click instead.
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFiles(Array.from(fileInput.files));
});

dropzone.addEventListener('click', () => {
  fileInput.value = ''; // reset so same file can be re-selected
  fileInput.click();
});

dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('over');
  handleFiles(Array.from(e.dataTransfer.files));
});

function handleFiles(files) {
  const max = state.modality === 'ctpet' ? 2 : 1;
  clearErr();
  if (files.length > max) {
    showErr(`${state.modality.toUpperCase()} expects ${max} file(s). You selected ${files.length}. Please select only ${max === 1 ? 'the CT file' : 'CT + PET files'}.`);
    log(`Too many files (max ${max} for ${state.modality.toUpperCase()})`, 'err');
    fileInput.value = '';
    fileList.innerHTML = '';
    state.files = [];
    runBtn.disabled = true;
    return;
  }
  state.files = files;
  fileList.innerHTML = '';
  files.forEach(f => {
    const chip = document.createElement('div');
    chip.className = 'file-chip';
    chip.textContent = `✓ ${f.name}`;
    fileList.appendChild(chip);
  });
  runBtn.disabled = files.length === 0;
  log(`${files.length}/${max} file(s) staged`, 'ok');
}

// ── View switcher (Raw / Mask / Overlay) ──────────
document.querySelectorAll('.view-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    switchView(btn.dataset.view);
  });
});

function switchView(v) {
  state.currentView = v;
  $('view-raw').style.display     = v === 'raw'     ? 'flex' : 'none';
  $('view-mask').style.display    = v === 'mask'    ? 'flex' : 'none';
  $('view-overlay').style.display = v === 'overlay' ? 'flex' : 'none';
}

// ── Build overlay on canvas ────────────────────────
function buildOverlay(rawSrc, maskSrc) {
  const rawImg  = new Image();
  const maskImg = new Image();
  let loaded = 0;

  const onLoad = () => {
    loaded++;
    if (loaded < 2) return;
    const W = rawImg.naturalWidth  || 512;
    const H = rawImg.naturalHeight || 512;
    overlayCanvas.width  = W;
    overlayCanvas.height = H;
    const ctx = overlayCanvas.getContext('2d');

    // Draw raw CT
    ctx.drawImage(rawImg, 0, 0, W, H);

    // Draw mask with colour tint + semi-transparency
    const offscreen = document.createElement('canvas');
    offscreen.width = W; offscreen.height = H;
    const octx = offscreen.getContext('2d');
    octx.drawImage(maskImg, 0, 0, W, H);

    // Colour the mask pixels teal
    const id = octx.getImageData(0, 0, W, H);
    for (let i = 0; i < id.data.length; i += 4) {
      const brightness = id.data[i]; // grayscale mask
      if (brightness > 10) {         // non-zero = tumour
        id.data[i]     = 0;          // R
        id.data[i + 1] = 220;        // G
        id.data[i + 2] = 240;        // B
        id.data[i + 3] = 160;        // alpha ~60%
      } else {
        id.data[i + 3] = 0;          // fully transparent
      }
    }
    octx.putImageData(id, 0, 0);
    ctx.drawImage(offscreen, 0, 0, W, H);
  };

  rawImg.crossOrigin  = 'anonymous';
  maskImg.crossOrigin = 'anonymous';
  rawImg.onload  = onLoad;
  maskImg.onload = onLoad;
  rawImg.src  = rawSrc;
  maskImg.src = maskSrc;
}

// ── Spinner helpers ────────────────────────────────
function showSpinner() {
  placeholder.style.display = 'flex';
  phRing.classList.add('spin');
  placeholder.querySelector('.ph-txt').textContent = '// processing…';
  $('view-raw').style.display     = 'none';
  $('view-mask').style.display    = 'none';
  $('view-overlay').style.display = 'none';
  viewControls.style.display = 'none';
  statsStrip.style.display   = 'none';
  dlRow.style.display        = 'none';
  imgLbl.textContent = 'PROCESSING…';
}

function hideSpinner() {
  phRing.classList.remove('spin');
  placeholder.style.display = 'none';
}

// ── Render segmentation result ─────────────────────
function renderSeg(data) {
  const base = API_BASE;

  // Raw slice preview (same endpoint but for the input slice)
  // Backend returns `preview` = mask preview; `raw_preview` = raw CT slice (we add this)
  const maskSrc = data.preview ? `${base}${data.preview}` : null;
  const rawSrc  = data.raw_preview ? `${base}${data.raw_preview}` : maskSrc; // fallback

  if (!maskSrc) { showErr('No preview image returned by server.'); return; }

  // Populate img elements
  $('img-raw').src  = rawSrc;
  $('img-mask').src = maskSrc;

  // Build overlay
  buildOverlay(rawSrc, maskSrc);

  // Show controls + default to overlay view if both available
  viewControls.style.display = 'flex';
  const defaultView = rawSrc !== maskSrc ? 'overlay' : 'mask';
  document.querySelectorAll('.view-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.view === defaultView);
  });
  switchView(defaultView);
  hideSpinner();

  // Stats
  if (data.stats) {
    $('sv').textContent   = data.stats.voxel_count?.toLocaleString() ?? '—';
    $('svol').textContent = data.stats.volume_cc != null ? data.stats.volume_cc.toFixed(1) : '—';
  }
  $('smod').textContent = 'nnU-Net';
  statsStrip.style.display = 'flex';
  imgLbl.textContent = 'MASK — CENTER SLICE';

  // Downloads
  state.maskDownloadUrl = data.download ? `${base}${data.download}` : null;
  state.previewDownloadUrl = maskSrc;
  dlRow.style.display = 'flex';
}

// ── Render detection result ────────────────────────
function renderDet(data) {
  hideSpinner();
  imgLbl.textContent = 'DETECTION COMPLETE';

  // Switch to detection tab
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab[data-tab="detection"]').classList.add('active');
  $('panel-detection').classList.add('active');

  const imgs = data.inference_images || [];
  const dets = data.detections || [];

  detCount.textContent = dets.length;
  detCount.style.display = dets.length ? 'inline' : 'none';

  let html = '';

  if (imgs.length) {
    html += `<div class="sidebar-label" style="font-family:var(--mono);font-size:10px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-bottom:10px;">Annotated Slices</div>
    <div class="det-images">`;
    imgs.forEach(im => {
      const src = im.url.startsWith('http') ? im.url : `${API_BASE}${im.url}`;
      html += `<div class="det-img-thumb">
        <img src="${src}" alt="Detection" loading="lazy" />
        <div class="det-img-label">${im.image}</div>
      </div>`;
    });
    html += `</div>`;
  }

  if (dets.length) {
    html += `<div class="sidebar-label" style="margin-top:20px;font-family:var(--mono);font-size:10px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-bottom:10px;">Detection Table</div>
    <table class="det-table">
      <thead><tr>
        <th>Image</th><th>Class</th><th>Confidence</th><th>Bbox (x1,y1,x2,y2)</th>
      </tr></thead><tbody>`;
    dets.forEach(d => {
      const pct = (d.confidence * 100).toFixed(1);
      const w = Math.round(d.confidence * 80);
      html += `<tr>
        <td>${d.image}</td>
        <td>${d.class}</td>
        <td>${pct}% <span class="conf-bar" style="width:${w}px"></span></td>
        <td>${Math.round(d.x1)}, ${Math.round(d.y1)}, ${Math.round(d.x2)}, ${Math.round(d.y2)}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
  }

  if (!imgs.length && !dets.length) {
    html = `<div class="empty-state" style="flex:1;min-height:300px">
      <div class="icon">✅</div>
      <span>// no detections found</span>
    </div>`;
  }

  detContent.innerHTML = html;
}

// ── Render classification result ───────────────────
function renderCls(data) {
  hideSpinner();
  imgLbl.textContent = 'CLASSIFICATION COMPLETE';

  // Switch to classification tab
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab[data-tab="classification"]').classList.add('active');
  $('panel-classification').classList.add('active');

  const prob    = data.probability ?? 0;          // 0–1
  const pct     = Math.round(prob * 100);
  const label   = data.label ?? (prob > 0.5 ? 'Malignant' : 'Benign');
  const conf    = data.confidence ? (data.confidence * 100).toFixed(1) : pct;
  const riskCls = prob < 0.35 ? 'low' : prob < 0.65 ? 'mid' : 'high';
  const riskTxt = prob < 0.35 ? 'Low Risk' : prob < 0.65 ? 'Moderate Risk' : 'High Risk';
  const verdict = label === 'Malignant'
    ? 'The classifier indicates a high probability of malignancy. Clinical review and biopsy are strongly recommended.'
    : 'The classifier indicates a lower malignancy probability. Continued monitoring and clinical correlation are advised.';

  // SVG semi-circle gauge
  const R = 80, cx = 110, cy = 100;
  const startAngle = Math.PI;
  const sweep = Math.PI * prob;
  const ex = cx + R * Math.cos(Math.PI + sweep);
  const ey = cy + R * Math.sin(Math.PI + sweep);
  const largeArc = sweep > Math.PI / 2 ? 1 : 0;
  const trackPath = `M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`;
  const fillPath  = `M ${cx - R} ${cy} A ${R} ${R} 0 ${largeArc} 1 ${ex} ${ey}`;
  const fillColor = riskCls === 'low' ? '#00f5a0' : riskCls === 'mid' ? '#ffb830' : '#ff5c7a';

  clsBadge.textContent = riskTxt;
  clsBadge.style.display = 'inline';

  clsContent.innerHTML = `
    <div class="risk-gauge">
      <svg viewBox="0 0 220 110">
        <path d="${trackPath}" fill="none" stroke="#182330" stroke-width="14" stroke-linecap="round"/>
        <path d="${fillPath}"  fill="none" stroke="${fillColor}" stroke-width="14" stroke-linecap="round"/>
      </svg>
      <div class="risk-label">
        <div class="risk-pct ${riskCls}">${pct}%</div>
        <div class="risk-lbl">Malignancy Probability</div>
      </div>
    </div>
    <div class="risk-verdict">
      <h3>${label} — ${riskTxt}</h3>
      <p>${verdict}</p>
    </div>
    <div style="display:flex;gap:20px;font-family:var(--mono);font-size:11px;flex-wrap:wrap;justify-content:center;">
      <div><div style="color:var(--muted);font-size:9px;letter-spacing:2px;text-transform:uppercase;margin-bottom:3px">Probability</div><div style="color:${fillColor};font-size:20px;font-weight:700">${pct}%</div></div>
      <div><div style="color:var(--muted);font-size:9px;letter-spacing:2px;text-transform:uppercase;margin-bottom:3px">Confidence</div><div style="color:var(--accent);font-size:20px;font-weight:700">${conf}%</div></div>
      <div><div style="color:var(--muted);font-size:9px;letter-spacing:2px;text-transform:uppercase;margin-bottom:3px">Prediction</div><div style="color:#e8f4fa;font-size:20px;font-weight:700">${label}</div></div>
    </div>
    <div class="disclaimer">
      <strong>Research Use Only.</strong> This AI result is not a clinical diagnosis.
      Always consult a qualified radiologist or oncologist before making medical decisions.
    </div>`;
}

// ── Download buttons ───────────────────────────────
dlMask.addEventListener('click', () => {
  if (state.maskDownloadUrl) triggerDownload(state.maskDownloadUrl);
});
dlPreview.addEventListener('click', () => {
  if (state.previewDownloadUrl) triggerDownload(state.previewDownloadUrl);
});

function triggerDownload(url) {
  // Use fetch → blob → object URL to force download (works cross-origin)
  fetch(url)
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.blob();
    })
    .then(blob => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      // Derive filename from URL
      a.download = url.split('/').pop() || 'endoscan_output';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(a.href);
      log(`Downloaded: ${a.download}`, 'ok');
    })
    .catch(e => { log(`Download failed: ${e.message}`, 'err'); showErr(`Download failed: ${e.message}`); });
}

// ── Run inference ──────────────────────────────────
runBtn.addEventListener('click', async () => {
  clearErr();
  if (!state.files.length) return;

  runBtn.disabled = true;
  showSpinner();
  log(`Sending ${state.modality.toUpperCase()} / ${state.model} request…`, 'inf');

  try {
    const fd = new FormData();
    fd.append('modality', state.modality);
    fd.append('model_type', state.model);

    // Pre-read into Blobs to avoid postMessage cloning errors in iframe envs
    for (const f of state.files) {
      const buf  = await f.arrayBuffer();
      const blob = new Blob([buf], { type: 'application/octet-stream' });
      fd.append('files', blob, f.name);
    }

    const res  = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok || data.status === 'error') {
      throw new Error(data.detail || JSON.stringify(data));
    }

    log(`Job ${data.job_id} complete — model: ${data.model}`, 'ok');
    state.result = data;

    if (state.model === 'segmentation') {
      renderSeg(data);
    } else if (state.model === 'detection') {
      renderDet(data);
    } else {
      renderCls(data);
    }

  } catch (err) {
    log(`Error: ${err.message}`, 'err');
    showErr(err.message);
    hideSpinner();
    placeholder.style.display = 'flex';
    placeholder.querySelector('.ph-txt').textContent = '// error — see console';
    imgLbl.textContent = 'ERROR';
  } finally {
    runBtn.disabled = false;
  }
});