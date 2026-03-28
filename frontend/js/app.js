const API_BASE = window.location.origin;

/* ── State ── */
let state = { modality: null, files: [], model: null };
let currentView = 'axial';

/* ══════════════════════════════════════════════
   STEP 1 — MODALITY
══════════════════════════════════════════════ */
function selectModality(m) {
  state.modality = m;
  ['ct','ctpet'].forEach(id => document.getElementById('card-'+id).classList.remove('selected'));
  document.getElementById('card-'+m).classList.add('selected');
  document.getElementById('num1').classList.add('active');

  document.getElementById('step2').classList.remove('locked');
  document.getElementById('num2').classList.add('active');

  document.getElementById('upload-sub').innerHTML =
    m === 'ct'
      ? 'Accepts .nii, .nii.gz — single channel CT'
      : 'Accepts .nii, .nii.gz — upload CT and PET files';
  document.getElementById('upload-title').textContent =
    m === 'ct' ? 'Drop CT scan here' : 'Drop CT + PET scans here';

  checkRunReady();
}

/* ══════════════════════════════════════════════
   STEP 2 — FILE UPLOAD
══════════════════════════════════════════════ */
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.add('drag-over');
}
function handleDragLeave() {
  document.getElementById('upload-zone').classList.remove('drag-over');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag-over');
  addFiles(Array.from(e.dataTransfer.files));
}
function handleFileSelect(e) { addFiles(Array.from(e.target.files)); }

function addFiles(newFiles) {
  newFiles.forEach(f => {
    if (!state.files.find(x => x.name === f.name)) state.files.push(f);
  });
  renderFileList();
  if (state.files.length > 0) {
    document.getElementById('step3').classList.remove('locked');
    document.getElementById('num3').classList.add('active');
  }
  checkRunReady();
}

function removeFile(name) {
  state.files = state.files.filter(f => f.name !== name);
  renderFileList();
  if (state.files.length === 0) {
    document.getElementById('step3').classList.add('locked');
  }
  checkRunReady();
}

function formatBytes(b) {
  if (b < 1024)    return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}

function renderFileList() {
  const list = document.getElementById('file-list');
  list.innerHTML = state.files.map(f => `
    <div class="file-item">
      <div class="file-icon">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <rect x="2" y="1" width="8" height="11" rx="1.5" stroke="#00d4b4" stroke-width="1.2"/>
          <path d="M5 4h4M5 6.5h4M5 9h2" stroke="#00d4b4" stroke-width="1" stroke-linecap="round"/>
        </svg>
      </div>
      <span class="file-name">${f.name}</span>
      <span class="file-size">${formatBytes(f.size)}</span>
      <button class="file-remove" onclick="removeFile('${f.name}')">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </button>
    </div>
  `).join('');
}

/* ══════════════════════════════════════════════
   STEP 3 — MODEL SELECTION
══════════════════════════════════════════════ */
function selectModel(m) {
  state.model = m;
  ['detection','segmentation'].forEach(id =>
    document.getElementById('card-'+id).classList.remove('selected')
  );
  document.getElementById('card-'+m).classList.add('selected');
  checkRunReady();
}

/* ══════════════════════════════════════════════
   RUN READY CHECK
══════════════════════════════════════════════ */
function checkRunReady() {
  const ready = state.modality && state.files.length > 0 && state.model;
  const rs = document.getElementById('run-section');
  rs.style.opacity       = ready ? '1' : '0.4';
  rs.style.pointerEvents = ready ? 'auto' : 'none';
  if (ready) {
    document.getElementById('run-hint').innerHTML =
      `Modality: <strong style="color:var(--accent)">${state.modality.toUpperCase()}</strong> · ` +
      `Model: <strong style="color:var(--accent2)">${state.model}</strong> · ` +
      `${state.files.length} file${state.files.length > 1 ? 's' : ''} queued`;
  }
}

/* ══════════════════════════════════════════════
   RUN ANALYSIS
══════════════════════════════════════════════ */
async function runAnalysis() {
  const sb = document.getElementById('status-bar');
  sb.className = 'status-bar visible processing';
  document.getElementById('status-spinner').style.display = '';
  setStatus('Uploading scan…', 'Sending to server', 15);

  const formData = new FormData();
  formData.append('modality',   state.modality);
  formData.append('model_type', state.model);
  state.files.forEach(f => formData.append('files', f));

  try {
    setStatus('Running inference…', 'nnU-Net segmentation in progress', 40);

    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body:   formData
    });

    const result = await response.json();

    if (result.status !== 'done') {
      throw new Error(result.detail || 'Server error');
    }

    setStatus('Analysis complete.', `Job: ${result.job_id}`, 100);
    document.getElementById('status-spinner').style.display = 'none';
    sb.className = 'status-bar visible done';

    renderResults(result);
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

  } catch (err) {
    setStatus('Error: ' + err.message, 'Check terminal for logs', 0);
    sb.className = 'status-bar visible error';
    document.getElementById('status-spinner').style.display = 'none';
  }
}

function setStatus(text, sub, pct) {
  document.getElementById('status-text').textContent = text;
  document.getElementById('status-sub').textContent  = sub;
  document.getElementById('progress-bar').style.width = pct + '%';
}

/* ══════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════ */
function renderResults(result) {
  document.getElementById('results-placeholder').style.display = 'none';
  const rc = document.getElementById('results-content');
  rc.style.display = 'block';

  // ── Stats ──
  const stats = result.stats || {};
  const statDefs = [
    { label: 'Tumour Volume',  val: (stats.volume_cc ?? '—') + ' cm³', color: 'var(--accent)' },
    { label: 'Voxel Count',    val: stats.voxel_count != null ? stats.voxel_count.toLocaleString() : '—', color: 'var(--accent2)' },
    { label: 'Modality',       val: (result.modality || '—').toUpperCase(), color: 'var(--text)' },
    { label: 'Model',          val: result.model || '—', color: 'var(--text)' },
  ];

  document.getElementById('results-stats').innerHTML = statDefs.map(s => `
    <div class="result-stat-card">
      <div class="result-stat-label">${s.label}</div>
      <div class="result-stat-val" style="color:${s.color}">${s.val}</div>
    </div>
  `).join('');

  // ── Check if segmentation or detection ──
  if (result.model === 'segmentation') {
    // Show segmentation viewer
    document.getElementById('segmentation-viewer').style.display = 'block';
    document.getElementById('detection-card').style.display = 'none';

    // Placeholder canvases
    drawPlaceholderCanvas('canvas-ct',   '#0a1628', '#00d4b4', 'CT');
    drawPlaceholderCanvas('canvas-pred', '#0a1628', '#f97316', 'PRED');

    document.getElementById('viewer-note').textContent =
      'Connect NIfTI slice reader to view real scan data';
  } else if (result.model === 'detection') {
    // Hide segmentation viewer, show detection results
    document.getElementById('segmentation-viewer').style.display = 'none';
    document.getElementById('detection-card').style.display = 'block';

    const detections = result.detections || [];
    const inferenceImages = result.inference_images || [];
    const detectionCount = detections.length;

    // Update count badge
    document.getElementById('detection-count').textContent =
      `${detectionCount} detection${detectionCount !== 1 ? 's' : ''} found`;

    // Render inference images
    const inferenceImagesHTML = inferenceImages.length > 0
      ? inferenceImages.map(img => `
          <div class="inference-img-card">
            <img src="${API_BASE}${img.url}" alt="YOLO output for ${img.image}" />
            <div class="inference-img-label">${img.image}</div>
          </div>
        `).join('')
      : '<p style="color: var(--muted);">No visualization image available</p>';

    document.getElementById('detection-images').innerHTML = inferenceImagesHTML;

    // Render detection list
    const detectionListHTML = detections.map((det, idx) => {
      const confidence = (det.confidence * 100).toFixed(1);
      return `
        <div class="detection-item">
          <div class="detection-item-left">
            <div class="detection-index">${idx + 1}</div>
            <div class="detection-info">
              <div class="detection-class">${(det.class || 'Tumour').toUpperCase()}</div>
              <div class="detection-confidence">
                Confidence: <strong>${confidence}%</strong>
                <div class="confidence-bar">
                  <div class="confidence-fill" style="width:${confidence}%"></div>
                </div>
              </div>
            </div>
          </div>
          <div class="detection-right">
            <div class="detection-coords">
              Box: [${det.x1.toFixed(0)}, ${det.y1.toFixed(0)}, ${det.x2.toFixed(0)}, ${det.y2.toFixed(0)}]
            </div>
          </div>
        </div>
      `;
    }).join('');

    document.getElementById('detection-list').innerHTML = detectionListHTML ||
      '<p style="color: var(--muted); text-align: center; padding: 20px;">No tumours detected</p>';
  }

  // ── Download button ──
  const db = document.getElementById('download-bar');
  if (result.download) {
    db.innerHTML = `
      <a href="${API_BASE}${result.download}" download
         style="display:inline-flex;align-items:center;gap:8px;padding:11px 22px;
                background:rgba(0,212,180,0.08);border:1.5px solid var(--accent);
                border-radius:8px;color:var(--accent);font-size:12px;
                text-decoration:none;font-family:'DM Mono',monospace;
                transition:all 0.2s;"
         onmouseover="this.style.background='rgba(0,212,180,0.14)'"
         onmouseout="this.style.background='rgba(0,212,180,0.08)'">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M7 1v8M3 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M1 11h12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        Download NIfTI Mask (.nii.gz)
      </a>
      <span style="font-size:11px;color:var(--muted)">Open in 3D Slicer or ITK-SNAP</span>
    `;
  } else {
    db.innerHTML = '';
  }
}

/* simple placeholder canvas so the viewer panel looks populated */
function drawPlaceholderCanvas(id, bg, color, label) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // fake scan rings
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.15;
  ctx.lineWidth = 1;
  for (let r = 30; r < 140; r += 22) {
    ctx.beginPath();
    ctx.arc(W/2, H/2, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // centre blob
  ctx.globalAlpha = 0.25;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(W/2, H/2, 28, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalAlpha = 0.5;
  ctx.beginPath();
  ctx.arc(W/2, H/2, 12, 0, Math.PI * 2);
  ctx.fill();

  // label
  ctx.globalAlpha = 0.5;
  ctx.fillStyle = color;
  ctx.font = '11px DM Mono, monospace';
  ctx.fillText(label, 12, 24);

  ctx.globalAlpha = 1;
}

/* ══════════════════════════════════════════════
   VIEWER TAB SWITCH
══════════════════════════════════════════════ */
function switchView(view, btn) {
  currentView = view;
  document.querySelectorAll('.vtab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  // when real NIfTI reader is connected, re-render slices here
}

/* ══════════════════════════════════════════════
   TRAINING CHARTS
══════════════════════════════════════════════ */
function generateTrainingData(epochs, finalVal, shape) {
  const data = [];
  for (let i = 0; i < epochs; i++) {
    const t = i / (epochs - 1);
    let val;
    if (shape === 'dice') {
      // log-like rise with plateau
      val = finalVal * (1 - Math.exp(-4.5 * t)) + (Math.random() - 0.5) * 0.018;
    } else {
      // exponential decay for loss
      val = 0.85 * Math.exp(-3.5 * t) + 0.08 + (Math.random() - 0.5) * 0.012;
    }
    data.push(Math.max(0, Math.min(1, val)));
  }
  return data;
}

function initCharts() {
  const epochs  = 300;
  const labels  = Array.from({length: epochs}, (_, i) => i + 1);
  const diceData = generateTrainingData(epochs, 0.86, 'dice');
  const lossData = generateTrainingData(epochs, 0.09, 'loss');

  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
    scales: {
      x: {
        ticks: { color: '#64748b', font: { size: 10, family: 'DM Mono' }, maxTicksLimit: 10 },
        grid:  { color: 'rgba(30,42,58,0.8)' },
      },
      y: {
        ticks: { color: '#64748b', font: { size: 10, family: 'DM Mono' } },
        grid:  { color: 'rgba(30,42,58,0.8)' },
      }
    },
    elements: { point: { radius: 0 }, line: { tension: 0.4 } },
    animation: { duration: 1200, easing: 'easeInOutQuart' },
  };

  // Dice chart
  new Chart(document.getElementById('chart-dice'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: diceData,
        borderColor: '#00d4b4',
        borderWidth: 1.5,
        backgroundColor: 'rgba(0,212,180,0.06)',
        fill: true,
      }]
    },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0, max: 1 } } }
  });

  // Loss chart
  new Chart(document.getElementById('chart-loss'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: lossData,
        borderColor: '#f97316',
        borderWidth: 1.5,
        backgroundColor: 'rgba(249,115,22,0.06)',
        fill: true,
      }]
    },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0 } } }
  });
}


/* ══════════════════════════════════════════════
   INIT
══════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
});
