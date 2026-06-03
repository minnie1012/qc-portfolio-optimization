// QC Portfolio Optimization dashboard

const PLOTLY_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#e6edf6', family: 'Inter, -apple-system, sans-serif', size: 12 },
  margin: { l: 50, r: 20, t: 46, b: 50 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' },
  yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' },
  // Legend above the plot so it never collides with x-axis ticks/titles.
  legend: { orientation: 'h', y: 1.1, yanchor: 'bottom', x: 0, font: { size: 10 } },
};

const PLOTLY_CFG = { displaylogo: false, responsive: true };

const ALGO_COLORS = {
  qsw: '#7cc4ff',
  warm_start_qaoa_saksham: '#b794ff',
  qaoa: '#5cd6a3',
  cvar_vqe: '#ffb86b',
  brute_force: '#ff7a7a',
  simulated_annealing: '#f6d56b',
};
const colorFor = (algo) => ALGO_COLORS[algo] || '#8a97ad';

const ALGO_DISPLAY = {
  warm_start_qaoa_saksham: 'warm_start_qaoa',
};
const displayAlgo = (algo) => ALGO_DISPLAY[algo] || algo;

const state = {
  instances: [],
  results: [],
  currentInstance: null,
  currentInstanceData: null,
  currentResults: [],
  algoFilter: '',
  bucketFilter: '',
  table: null,
  comparison: null,
  cmpInstance: '',
  cmpTable: null,
};

const TYPE_COLOR = { classical: '#ffb86b', quantum: '#7cc4ff' };

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

function fmtNum(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (typeof v !== 'number') return String(v);
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(2);
  return v.toFixed(digits);
}

function feasibilityTag(f) {
  if (f === true) return '<span class="tag tag-good">yes</span>';
  if (f === false) return '<span class="tag tag-bad">no</span>';
  return '—';
}

async function init() {
  const [instances, results, comparison] = await Promise.all([
    fetchJSON('/api/instances'),
    fetchJSON('/api/results'),
    fetchJSON('/api/comparison'),
  ]);
  state.instances = instances;
  state.results = results;
  state.comparison = comparison;

  populateKPIs();
  populateControls();
  buildTable();
  drawScatter();
  setupComparison();
  initQuantum(instances);

  if (instances.length) {
    document.getElementById('instance-select').value = instances[0].instance_id;
    await selectInstance(instances[0].instance_id);
  }
}

function populateKPIs() {
  const algos = new Set(state.results.map((r) => r.algorithm));
  const feasibleCount = state.results.filter((r) => r.feasible === true).length;
  document.getElementById('kpi-instances').textContent = state.instances.length;
  document.getElementById('kpi-runs').textContent = state.results.length;
  document.getElementById('kpi-algos').textContent = algos.size;
  const pct = state.results.length
    ? Math.round((feasibleCount / state.results.length) * 100)
    : 0;
  document.getElementById('kpi-feasible').textContent = `${pct}%`;
}

function populateControls() {
  const instSel = document.getElementById('instance-select');
  instSel.innerHTML = '';
  state.instances.forEach((inst) => {
    const opt = document.createElement('option');
    opt.value = inst.instance_id;
    opt.textContent = `${inst.instance_id}  ·  N=${inst.N}, K=${inst.K}`;
    instSel.appendChild(opt);
  });
  instSel.addEventListener('change', (e) => selectInstance(e.target.value));

  const algos = Array.from(new Set(state.results.map((r) => r.algorithm))).sort();
  const algoSel = document.getElementById('algo-filter');
  algos.forEach((a) => {
    const opt = document.createElement('option');
    opt.value = a;
    opt.textContent = displayAlgo(a);
    algoSel.appendChild(opt);
  });
  algoSel.addEventListener('change', (e) => {
    state.algoFilter = e.target.value;
    refreshTable();
    drawScatter();
  });

  const buckets = Array.from(new Set(state.instances.map((i) => i.bucket))).sort();
  const bSel = document.getElementById('bucket-filter');
  buckets.forEach((b) => {
    const opt = document.createElement('option');
    opt.value = b;
    opt.textContent = b;
    bSel.appendChild(opt);
  });
  bSel.addEventListener('change', (e) => {
    state.bucketFilter = e.target.value;
    refreshTable();
    drawScatter();
  });
}

function instanceBucket(instance_id) {
  const inst = state.instances.find((i) => i.instance_id === instance_id);
  return inst ? inst.bucket : 'other';
}

function filteredResults() {
  return state.results.filter((r) => {
    if (state.algoFilter && r.algorithm !== state.algoFilter) return false;
    if (state.bucketFilter && instanceBucket(r.instance_id) !== state.bucketFilter) return false;
    return true;
  });
}

async function selectInstance(instanceId) {
  state.currentInstance = instanceId;
  const [inst, results] = await Promise.all([
    fetchJSON(`/api/instance/${instanceId}`),
    fetchJSON(`/api/results_for/${instanceId}`),
  ]);
  state.currentInstanceData = inst;
  state.currentResults = results;
  renderInstanceMeta(inst);
  drawMu(inst);
  drawSigma(inst);
  drawWeights(inst, results);
}

function renderInstanceMeta(inst) {
  const el = document.getElementById('instance-meta');
  const dateRange = inst.date_range ? inst.date_range.join(' → ') : '—';
  el.innerHTML = `
    <div><strong>${inst.instance_id}</strong> · N=${inst.N}, K=${inst.K}, q=${fmtNum(inst.q, 3)}</div>
    <div>Window: ${dateRange}</div>
    <div>Tickers: ${inst.asset_tickers.join(', ')}</div>
  `;
}

function drawMu(inst) {
  const trace = {
    type: 'bar',
    x: inst.asset_tickers,
    y: inst.mu,
    marker: {
      color: inst.mu.map((v) => (v >= 0 ? '#5cd6a3' : '#ff7a7a')),
    },
    hovertemplate: '%{x}<br>μ = %{y:.4f}<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'annualized return' },
  };
  Plotly.newPlot('chart-mu', [trace], layout, PLOTLY_CFG);
}

function drawSigma(inst) {
  const trace = {
    type: 'heatmap',
    z: inst.sigma,
    x: inst.asset_tickers,
    y: inst.asset_tickers,
    colorscale: 'Viridis',
    hovertemplate: '%{x} ↔ %{y}<br>Σ = %{z:.4f}<extra></extra>',
    colorbar: { thickness: 10, len: 0.9 },
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -45 },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, autorange: 'reversed' },
  };
  Plotly.newPlot('chart-sigma', [trace], layout, PLOTLY_CFG);
}

function normalizeWeights(bitstring) {
  if (!bitstring) return null;
  const arr = bitstring.map((v) => Number(v) || 0);
  const sum = arr.reduce((a, b) => a + b, 0);
  if (sum <= 0) return arr;
  // If integer 0/1 binary, leave as-is; else normalize so it sums to 1.
  const isBinary = arr.every((v) => v === 0 || v === 1);
  if (isBinary) return arr;
  return arr.map((v) => v / sum);
}

function drawWeights(inst, results) {
  const traces = results
    .filter((r) => Array.isArray(r.bitstring) && r.bitstring.length === inst.N)
    .map((r) => ({
      type: 'bar',
      name: displayAlgo(r.algorithm),
      x: inst.asset_tickers,
      y: normalizeWeights(r.bitstring),
      marker: { color: colorFor(r.algorithm) },
      hovertemplate: `${displayAlgo(r.algorithm)}<br>%{x}: %{y:.3f}<extra></extra>`,
    }));
  if (!traces.length) {
    Plotly.purge('chart-weights');
    document.getElementById('chart-weights').innerHTML =
      '<p style="color:#8a97ad; padding:24px;">No results with weight vectors for this instance.</p>';
    return;
  }
  const layout = {
    ...PLOTLY_LAYOUT,
    barmode: 'group',
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'allocation', rangemode: 'tozero' },
  };
  Plotly.newPlot('chart-weights', traces, layout, PLOTLY_CFG);
}

function drawScatter() {
  const rows = filteredResults();
  const grouped = {};
  rows.forEach((r) => {
    if (r.wall_time_seconds == null || r.objective_value == null) return;
    (grouped[r.algorithm] = grouped[r.algorithm] || []).push(r);
  });
  const traces = Object.entries(grouped).map(([algo, items]) => ({
    type: 'scatter',
    mode: 'markers',
    name: displayAlgo(algo),
    x: items.map((r) => r.wall_time_seconds),
    y: items.map((r) => r.objective_value),
    text: items.map((r) => `${r.instance_id}<br>feasible: ${r.feasible}`),
    hovertemplate: `${displayAlgo(algo)}<br>%{text}<br>time=%{x:.3f}s<br>obj=%{y:.4f}<extra></extra>`,
    marker: {
      size: 9,
      color: colorFor(algo),
      line: { color: 'rgba(255,255,255,0.25)', width: 0.5 },
    },
  }));
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'wall time (s)', type: 'log' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'objective value' },
  };
  Plotly.newPlot('chart-scatter', traces, layout, PLOTLY_CFG);
}

function buildTable() {
  state.table = $('#runs-table').DataTable({
    data: tableRows(),
    pageLength: 15,
    order: [[3, 'asc']],
    columnDefs: [
      { targets: [3, 5, 6], className: 'dt-right' },
      { targets: [7, 8, 9, 10, 11], className: 'dt-right' },
    ],
  });
}

function refreshTable() {
  if (!state.table) return;
  state.table.clear();
  state.table.rows.add(tableRows());
  state.table.draw();
}

function tableRows() {
  return filteredResults().map((r) => [
    displayAlgo(r.algorithm),
    r.instance_id,
    `<span class="tag tag-bucket">${instanceBucket(r.instance_id)}</span>`,
    fmtNum(r.objective_value, 5),
    feasibilityTag(r.feasible),
    fmtNum(r.approx_ratio, 3),
    fmtNum(r.wall_time_seconds, 4),
    r.qubit_count ?? '—',
    r.circuit_depth ?? '—',
    r.two_qubit_gate_count ?? '—',
    r.optimizer_iters ?? '—',
    r.shots ?? '—',
  ]);
}

// --- Quantum vs. Classical ---------------------------------------------------

function setupComparison() {
  const cmp = state.comparison;
  if (!cmp || !cmp.records.length) return;

  document.getElementById('cmp-instance-list').textContent =
    cmp.instances.join(', ');

  const sel = document.getElementById('cmp-instance');
  cmp.instances.forEach((iid) => {
    const opt = document.createElement('option');
    opt.value = iid;
    opt.textContent = iid;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', (e) => {
    state.cmpInstance = e.target.value;
    drawComparison();
  });

  buildCmpTable();
  drawComparison();
}

function cmpRecords() {
  const recs = state.comparison.records;
  if (!state.cmpInstance) return recs;
  return recs.filter((r) => r.instance_id === state.cmpInstance);
}

// Aggregate to one value per (solver) for bar charts: mean across instances/runs.
function aggregateBy(records, field) {
  const groups = {};
  records.forEach((r) => {
    if (r[field] == null) return;
    const key = `${r.solver}|${r.type}`;
    (groups[key] = groups[key] || []).push(r[field]);
  });
  return Object.entries(groups)
    .map(([key, vals]) => {
      const [solver, type] = key.split('|');
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      return { solver, type, value: mean };
    })
    .sort((a, b) => a.value - b.value);
}

function drawComparison() {
  const recs = cmpRecords();
  drawCmpObjective(recs);
  drawCmpWalltime(recs);
  drawCmpScatter(recs);
  refreshCmpTable(recs);
}

function drawCmpObjective(recs) {
  const agg = aggregateBy(recs, 'objective_value');
  const trace = {
    type: 'bar',
    x: agg.map((d) => displayAlgo(d.solver)),
    y: agg.map((d) => d.value),
    marker: { color: agg.map((d) => TYPE_COLOR[d.type]) },
    hovertemplate: '%{x}<br>objective = %{y:.4f}<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    margin: { ...PLOTLY_LAYOUT.margin, b: 110 },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -35, automargin: true },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'objective value' },
    showlegend: false,
  };
  Plotly.newPlot('chart-cmp-objective', [trace], layout, PLOTLY_CFG);
}

function drawCmpWalltime(recs) {
  const agg = aggregateBy(recs, 'wall_time_seconds').sort((a, b) => a.value - b.value);
  const trace = {
    type: 'bar',
    x: agg.map((d) => displayAlgo(d.solver)),
    y: agg.map((d) => d.value),
    marker: { color: agg.map((d) => TYPE_COLOR[d.type]) },
    hovertemplate: '%{x}<br>%{y:.4f}s<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    margin: { ...PLOTLY_LAYOUT.margin, b: 110 },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -35, automargin: true },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'wall time (s)', type: 'log' },
    showlegend: false,
  };
  Plotly.newPlot('chart-cmp-walltime', [trace], layout, PLOTLY_CFG);
}

function drawCmpScatter(recs) {
  const usable = recs.filter(
    (r) => r.expected_return != null && r.volatility != null,
  );
  const byType = { classical: [], quantum: [] };
  usable.forEach((r) => byType[r.type].push(r));

  const traces = Object.entries(byType)
    .filter(([, items]) => items.length)
    .map(([type, items]) => ({
      type: 'scatter',
      mode: 'markers',
      name: type,
      x: items.map((r) => r.volatility),
      y: items.map((r) => r.expected_return),
      text: items.map((r) => `${displayAlgo(r.solver)} · ${r.instance_id}`),
      hovertemplate:
        '%{text}<br>vol=%{x:.2f}%<br>return=%{y:.2f}%<extra></extra>',
      marker: {
        size: 13,
        symbol: type === 'quantum' ? 'diamond' : 'circle',
        color: TYPE_COLOR[type],
        line: { color: 'rgba(255,255,255,0.4)', width: 1 },
      },
    }));

  // Constant-Sharpe reference lines.
  const maxVol = Math.max(10, ...usable.map((r) => r.volatility));
  const shapes = [0.5, 1.0, 1.5].map((s) => ({
    type: 'line',
    x0: 0,
    y0: 0,
    x1: maxVol,
    y1: s * maxVol,
    line: { color: 'rgba(255,255,255,0.15)', width: 1, dash: 'dash' },
  }));
  const annotations = [0.5, 1.0, 1.5].map((s) => ({
    x: maxVol,
    y: s * maxVol,
    text: `Sharpe ${s}`,
    showarrow: false,
    font: { size: 10, color: 'rgba(255,255,255,0.4)' },
    xanchor: 'right',
    yanchor: 'bottom',
  }));

  if (!traces.length) {
    Plotly.purge('chart-cmp-scatter');
    document.getElementById('chart-cmp-scatter').innerHTML =
      '<p style="color:#8a97ad; padding:24px;">No solvers report return/volatility for this selection.</p>';
    return;
  }

  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'volatility %', rangemode: 'tozero' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'expected return %', rangemode: 'tozero' },
    shapes,
    annotations,
  };
  Plotly.newPlot('chart-cmp-scatter', traces, layout, PLOTLY_CFG);
}

function typeTag(type) {
  const cls = type === 'quantum' ? 'tag-bucket' : 'tag-good';
  return `<span class="tag ${cls}">${type}</span>`;
}

function cmpTableRows(recs) {
  return recs.map((r) => [
    displayAlgo(r.solver),
    typeTag(r.type),
    r.instance_id,
    fmtNum(r.objective_value, 4),
    fmtNum(r.expected_return, 2),
    fmtNum(r.volatility, 2),
    fmtNum(r.sharpe, 3),
    fmtNum(r.wall_time_seconds, 4),
    feasibilityTag(r.feasible),
  ]);
}

function buildCmpTable() {
  state.cmpTable = $('#cmp-table').DataTable({
    data: cmpTableRows(cmpRecords()),
    pageLength: 12,
    order: [[6, 'desc']],
  });
}

function refreshCmpTable(recs) {
  if (!state.cmpTable) return;
  state.cmpTable.clear();
  state.cmpTable.rows.add(cmpTableRows(recs));
  state.cmpTable.draw();
}

// ===========================================================================
// Quantum showcase: HHL scaling + linear solve, QSW open-quantum-walk dynamics
// ===========================================================================

const QSW_PRESETS = {
  moderate_balanced: { label: 'Moderate-Balanced', alpha: 10, beta: 10, lam: 10 },
  ultra_diversified: { label: 'Ultra-Diversified', alpha: 1, beta: 100, lam: 10 },
  stability_focused: { label: 'Stability-Focused', alpha: 1, beta: 10, lam: 100 },
  balanced_active: { label: 'Balanced-Active', alpha: 10, beta: 1, lam: 100 },
  sharpe_maximizer: { label: 'Sharpe-Maximizer', alpha: 100, beta: 1, lam: 10 },
};

const QSW_MAX_N = 24;

const qstate = {
  instanceId: null,
  evolution: null,
  steady: null,
  hhl: null,
  timeIdx: 0,
  omega: 0.1,
  preset: 'moderate_balanced',
  playing: false,
  playTimer: null,
  loadToken: 0,
};

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function initQuantum(instances) {
  // The QSW Lindblad solve is n²×n²; cap the selector to keep it interactive.
  const usable = instances.filter((i) => i.N <= QSW_MAX_N)
    .sort((a, b) => a.N - b.N || a.instance_id.localeCompare(b.instance_id));
  const sel = document.getElementById('quantum-instance');
  usable.forEach((inst) => {
    const opt = document.createElement('option');
    opt.value = inst.instance_id;
    opt.textContent = `${inst.instance_id}  ·  N=${inst.N}, K=${inst.K}`;
    sel.appendChild(opt);
  });
  // Prefer a small (not tiny) instance so the walk has room to spread.
  const preferred = usable.find((i) => i.N >= 8) || usable[0];
  if (!preferred) return;
  qstate.instanceId = preferred.instance_id;
  sel.value = preferred.instance_id;
  sel.addEventListener('change', (e) => {
    qstate.instanceId = e.target.value;
    loadHHL();
    loadQSW();
  });

  // Preset dropdown
  const psel = document.getElementById('qsw-preset');
  Object.entries(QSW_PRESETS).forEach(([key, p]) => {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = p.label;
    psel.appendChild(opt);
  });
  psel.value = qstate.preset;
  psel.addEventListener('change', (e) => {
    qstate.preset = e.target.value;
    loadQSW();
  });

  // HHL scaling sliders (pure-JS, instant)
  ['hhl-n', 'hhl-kappa', 'hhl-eps'].forEach((id) => {
    document.getElementById(id).addEventListener('input', drawHHLScaling);
  });

  // QSW omega slider (re-solves on the server, debounced)
  const omegaSlider = document.getElementById('qsw-omega');
  const omegaDebounced = debounce(loadQSW, 110);
  omegaSlider.addEventListener('input', (e) => {
    qstate.omega = Number(e.target.value) / 100;
    document.getElementById('qsw-omega-val').textContent = qstate.omega.toFixed(2);
    omegaDebounced();
  });

  // Time scrubber (local — no fetch)
  document.getElementById('qsw-time').addEventListener('input', (e) => {
    qstate.timeIdx = Number(e.target.value);
    renderQSWTime();
  });

  document.getElementById('qsw-play').addEventListener('click', togglePlay);

  drawHHLScaling();
  loadHHL();
  loadQSW();
}

// --- HHL complexity scaling (asymptotic operation counts) -------------------

function hhlScalingParams() {
  const nExp = Number(document.getElementById('hhl-n').value);     // 1..20
  const kappa = Number(document.getElementById('hhl-kappa').value); // 1..1000
  const epsExp = Number(document.getElementById('hhl-eps').value);  // 1..4
  const eps = Math.pow(10, -epsExp);
  const N = Math.pow(2, nExp);
  document.getElementById('hhl-n-val').textContent = N.toLocaleString();
  document.getElementById('hhl-kappa-val').textContent = kappa;
  document.getElementById('hhl-eps-val').textContent = eps.toString();
  return { nExp, N, kappa, eps };
}

// Asymptotic proxies (arbitrary units): the slopes & crossover are the point.
const costDirect = (N) => Math.pow(N, 3);                         // classical LU  O(N³)
const costCG = (N, k, e) => N * k * Math.log2(1 / e + 1);        // classical CG  O(N·κ·log 1/ε)
const costHHL = (N, k, e) => Math.log2(N + 1) * k * k / e;        // HHL  O(log N·κ²/ε)

function drawHHLScaling() {
  const { nExp, N, kappa, eps } = hhlScalingParams();
  const xs = [];
  for (let e = 1; e <= 20; e += 1) xs.push(Math.pow(2, e));
  const yDirect = xs.map((n) => costDirect(n));
  const yCG = xs.map((n) => costCG(n, kappa, eps));
  const yHHL = xs.map((n) => costHHL(n, kappa, eps));

  const mk = (name, x, y, color, dash) => ({
    type: 'scatter', mode: 'lines', name, x, y,
    line: { color, width: 2.4, dash: dash || 'solid' },
    hovertemplate: `${name}<br>N=%{x}<br>ops≈%{y:.3g}<extra></extra>`,
  });
  const traces = [
    mk('Classical — direct O(N³)', xs, yDirect, '#ff7a7a', 'dot'),
    mk('Classical — CG O(N·κ·log 1/ε)', xs, yCG, '#ffb86b'),
    mk('HHL — quantum O(log N·κ²/ε)', xs, yHHL, '#7cc4ff'),
  ];

  // Marker at the selected N
  const cgN = costCG(N, kappa, eps);
  const hhlN = costHHL(N, kappa, eps);
  traces.push({
    type: 'scatter', mode: 'markers', name: `N = ${N.toLocaleString()}`,
    x: [N, N], y: [cgN, hhlN], showlegend: false,
    marker: { size: 11, color: ['#ffb86b', '#7cc4ff'], line: { color: '#fff', width: 1.2 } },
    hovertemplate: 'N=%{x}<br>ops≈%{y:.3g}<extra></extra>',
  });

  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'assets N', type: 'log' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'operations (asymptotic, a.u.)', type: 'log' },
    shapes: [{
      type: 'line', x0: N, x1: N, y0: Math.min(cgN, hhlN), y1: Math.max(cgN, hhlN),
      line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dash' },
    }],
  };
  Plotly.react('chart-hhl-scaling', traces, layout, PLOTLY_CFG);

  // Headline speedup in the banner
  const speedup = cgN / hhlN;
  const el = document.getElementById('qa-speedup');
  if (speedup >= 1) {
    el.textContent = `${speedup >= 100 ? Math.round(speedup) : speedup.toFixed(1)}×`;
    el.style.color = 'var(--good)';
  } else {
    el.textContent = `${(1 / speedup).toFixed(1)}× ↓`;
    el.style.color = 'var(--warn)';
  }
  document.querySelector('#qa-stats .qa-stat-label').textContent =
    `HHL speedup @ N=${N.toLocaleString()}`;
}

// --- HHL on a real instance -------------------------------------------------

async function loadHHL() {
  const id = qstate.instanceId;
  if (!id) return;
  try {
    const data = await fetchJSON(`/api/quantum/hhl/${id}`);
    if (data.error) return;
    qstate.hhl = data;
    drawHHLWeights(data);
    fillHHLResource(data);
    fillHHLTable(data);
  } catch (e) {
    console.error('HHL load failed', e);
  }
}

function drawHHLWeights(d) {
  const sel = new Set(d.selected);
  const trace = {
    type: 'bar',
    x: d.tickers,
    y: d.weights,
    marker: {
      color: d.weights.map((_, i) => (sel.has(i) ? '#5cd6a3' : 'rgba(124,196,255,0.45)')),
      line: { color: d.weights.map((_, i) => (sel.has(i) ? '#5cd6a3' : 'rgba(124,196,255,0.3)')), width: 1 },
    },
    hovertemplate: '%{x}<br>w = %{y:.4f}<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'continuous weight wᵢ', zeroline: true },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -45 },
    showlegend: false,
    title: { text: `selected ${d.K} of ${d.N} assets (green)`, font: { size: 11, color: '#8a97ad' }, x: 0.02, y: 0.97 },
  };
  Plotly.react('chart-hhl-weights', [trace], layout, PLOTLY_CFG);
}

function fillHHLResource(d) {
  const r = d.resource;
  const chips = [
    ['qubits', r.qubit_count],
    ['circuit depth', r.circuit_depth.toLocaleString()],
    ['2-qubit gates', r.two_qubit_gate_count.toLocaleString()],
    ['n_b', r.n_b],
    ['clock qubits', r.n_clock],
    ['κ (cond.)', fmtNum(r.kappa, 1)],
  ];
  document.getElementById('hhl-resource').innerHTML =
    chips.map(([k, v]) => `<span class="chip">${k} <strong>${v}</strong></span>`).join('') +
    `<span class="chip">backend <strong>${d.backend.replace(/_/g, ' ')}</strong></span>`;
}

function fillHHLTable(d) {
  const sel = new Set(d.selected);
  const body = d.tickers.map((tk, i) => {
    const cls = sel.has(i) ? ' class="is-selected"' : '';
    return `<tr${cls}>
      <td>${i}</td><td>${tk}</td>
      <td>${fmtNum(d.mu[i], 4)}</td>
      <td>${fmtNum(d.weights[i], 4)}</td>
      <td>${fmtNum(d.weights_abs[i], 4)}</td>
      <td>${sel.has(i) ? '<span class="tag tag-good">selected</span>' : '—'}</td>
    </tr>`;
  }).join('');
  document.querySelector('#hhl-table tbody').innerHTML = body;
}

// --- QSW open-quantum-walk dynamics -----------------------------------------

async function loadQSW() {
  const id = qstate.instanceId;
  if (!id) return;
  const p = QSW_PRESETS[qstate.preset];
  const token = ++qstate.loadToken;
  const qp = `omega=${qstate.omega}&alpha=${p.alpha}&beta=${p.beta}&lam=${p.lam}`;
  try {
    const [evo, steady] = await Promise.all([
      fetchJSON(`/api/quantum/qsw_evolution/${id}?${qp}`),
      fetchJSON(`/api/quantum/qsw/${id}?${qp}`),
    ]);
    if (token !== qstate.loadToken) return; // stale (user kept sliding)
    if (evo.error || steady.error) {
      flagQSWUnavailable(evo.error || steady.error);
      return;
    }
    qstate.evolution = evo;
    qstate.steady = steady;
    qstate.timeIdx = Math.min(qstate.timeIdx, evo.times.length - 1);

    const tslider = document.getElementById('qsw-time');
    tslider.max = evo.times.length - 1;

    drawQSWHeatmap(evo);
    drawQSWParticipation(evo);
    drawQSWRho(evo);
    drawQSWWeights(steady);
    fillQSWStats(steady);
    fillQSWTable(steady);
    renderQSWTime();
    updateSpreadStat(evo);
  } catch (e) {
    console.error('QSW load failed', e);
  }
}

function flagQSWUnavailable(msg) {
  ['chart-qsw-heatmap', 'chart-qsw-participation', 'chart-qsw-snapshot',
    'chart-qsw-rho', 'chart-qsw-weights'].forEach((id) => {
    Plotly.purge(id);
    document.getElementById(id).innerHTML =
      `<p style="color:#8a97ad;padding:24px;">${msg}</p>`;
  });
}

// Build ~`count` evenly spaced, rounded tick labels for a numeric series.
function roundedTicks(values, count = 7, digits = 2) {
  const n = values.length;
  if (!n) return { tickvals: [], ticktext: [] };
  const step = Math.max(1, Math.round((n - 1) / (count - 1)));
  const tickvals = [];
  const ticktext = [];
  for (let i = 0; i < n; i += step) {
    tickvals.push(values[i]);
    ticktext.push(values[i].toFixed(digits));
  }
  return { tickvals, ticktext };
}

function drawQSWHeatmap(evo) {
  // z[asset][time] = population
  const N = evo.N;
  const z = [];
  for (let a = 0; a < N; a += 1) z.push(evo.population.map((row) => row[a]));
  const ticks = roundedTicks(evo.times);
  const trace = {
    type: 'heatmap',
    z, x: evo.times, y: evo.tickers,
    colorscale: 'Viridis',
    colorbar: { thickness: 10, len: 0.9, title: { text: 'pop', font: { size: 10 } } },
    hovertemplate: 't=%{x:.3f}<br>%{y}<br>pop=%{z:.3f}<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: {
      ...PLOTLY_LAYOUT.xaxis, title: 'time',
      tickmode: 'array', tickvals: ticks.tickvals, ticktext: ticks.ticktext,
    },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'asset' },
  };
  Plotly.react('chart-qsw-heatmap', [trace], layout, PLOTLY_CFG);
}

function drawQSWParticipation(evo) {
  const tQ = {
    type: 'scatter', mode: 'lines', name: `quantum (ω=${evo.omega.toFixed(2)})`,
    x: evo.times, y: evo.participation,
    line: { color: '#b794ff', width: 2.6 },
  };
  const tC = {
    type: 'scatter', mode: 'lines', name: 'classical (ω=1)',
    x: evo.times, y: evo.participation_classical,
    line: { color: '#ffb86b', width: 2, dash: 'dash' },
  };
  const ticks = roundedTicks(evo.times);
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: {
      ...PLOTLY_LAYOUT.xaxis, title: 'time',
      tickmode: 'array', tickvals: ticks.tickvals, ticktext: ticks.ticktext,
    },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'effective # assets occupied', rangemode: 'tozero' },
  };
  Plotly.react('chart-qsw-participation', [tQ, tC], layout, PLOTLY_CFG);
}

function drawQSWRho(evo) {
  const trace = {
    type: 'heatmap',
    z: evo.rho_peak, x: evo.tickers, y: evo.tickers,
    colorscale: 'Magma',
    colorbar: { thickness: 10, len: 0.9 },
    hovertemplate: '%{y} ↔ %{x}<br>|ρ| = %{z:.3f}<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -45 },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, autorange: 'reversed' },
  };
  Plotly.react('chart-qsw-rho', [trace], layout, PLOTLY_CFG);
}

function drawQSWWeights(steady) {
  const ew = steady.equal_weight;
  const trace = {
    type: 'bar', x: steady.tickers, y: steady.weights,
    marker: { color: '#7cc4ff' },
    hovertemplate: '%{x}<br>w = %{y:.4f}<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -45 },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'steady weight', rangemode: 'tozero' },
    showlegend: false,
    shapes: [{
      type: 'line', x0: -0.5, x1: steady.tickers.length - 0.5, y0: ew, y1: ew,
      line: { color: '#ff7a7a', width: 1.5, dash: 'dash' },
    }],
    annotations: [{
      x: steady.tickers.length - 0.5, y: ew, text: `1/N = ${ew.toFixed(3)}`,
      showarrow: false, font: { size: 10, color: '#ff7a7a' }, xanchor: 'right', yanchor: 'bottom',
    }],
  };
  Plotly.react('chart-qsw-weights', [trace], layout, PLOTLY_CFG);
}

function fillQSWStats(s) {
  const chips = [
    ['ω', s.omega.toFixed(2)],
    ['HHI', fmtNum(s.hhi, 3)],
    ['eff. # stocks', fmtNum(s.eff_stocks, 2)],
    ['coherence', fmtNum(s.coherence, 3)],
    ['N', s.N],
  ];
  document.getElementById('qsw-stats-strip').innerHTML =
    chips.map(([k, v]) => `<span class="chip">${k} <strong>${v}</strong></span>`).join('');
}

function fillQSWTable(s) {
  const ew = s.equal_weight;
  const body = s.tickers.map((tk, i) => {
    const w = s.weights[i];
    const rel = w - ew;
    const relTxt = `${rel >= 0 ? '+' : ''}${fmtNum(rel, 4)}`;
    const color = rel >= 0 ? 'var(--good)' : 'var(--bad)';
    return `<tr>
      <td>${i}</td><td>${tk}</td>
      <td>${fmtNum(w, 4)}</td>
      <td style="color:${color}">${relTxt}</td>
    </tr>`;
  }).join('');
  document.querySelector('#qsw-table tbody').innerHTML = body;
}

function renderQSWTime() {
  const evo = qstate.evolution;
  if (!evo) return;
  const i = Math.min(qstate.timeIdx, evo.times.length - 1);
  document.getElementById('qsw-time-val').textContent = evo.times[i].toFixed(3);

  // snapshot bar
  const trace = {
    type: 'bar', x: evo.tickers, y: evo.population[i],
    marker: { color: '#5cd6a3' },
    hovertemplate: '%{x}<br>pop = %{y:.3f}<extra></extra>',
  };
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -45 },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'population', range: [0, 1] },
    showlegend: false,
  };
  Plotly.react('chart-qsw-snapshot', [trace], layout, PLOTLY_CFG);

  // move the time marker on the heatmap
  const t = evo.times[i];
  Plotly.relayout('chart-qsw-heatmap', {
    shapes: [{
      type: 'line', x0: t, x1: t, y0: -0.5, y1: evo.N - 0.5,
      line: { color: '#ffffff', width: 2 },
    }],
  });
}

function updateSpreadStat(evo) {
  const last = evo.participation.length - 1;
  const q = evo.participation[last];
  const c = evo.participation_classical[last];
  const ratio = c > 1e-9 ? q / c : 1;
  const el = document.getElementById('qa-spread');
  el.textContent = `${ratio.toFixed(2)}×`;
  el.style.color = ratio >= 1 ? 'var(--good)' : 'var(--warn)';
}

function togglePlay() {
  const btn = document.getElementById('qsw-play');
  if (qstate.playing) {
    clearInterval(qstate.playTimer);
    qstate.playing = false;
    btn.classList.remove('playing');
    btn.textContent = '▶ Animate walk';
    return;
  }
  if (!qstate.evolution) return;
  qstate.playing = true;
  btn.classList.add('playing');
  btn.textContent = '⏸ Pause';
  const slider = document.getElementById('qsw-time');
  qstate.playTimer = setInterval(() => {
    const n = qstate.evolution.times.length;
    qstate.timeIdx = (qstate.timeIdx + 1) % n;
    slider.value = qstate.timeIdx;
    renderQSWTime();
  }, 120);
}

init().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    'afterbegin',
    `<div style="padding:14px; background:#3a1a1a; color:#ffb;">Failed to load: ${err.message}</div>`,
  );
});
