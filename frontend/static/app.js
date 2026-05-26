// QC Portfolio Optimization dashboard

const PLOTLY_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#e6edf6', family: 'Inter, -apple-system, sans-serif', size: 12 },
  margin: { l: 50, r: 20, t: 30, b: 50 },
  xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' },
  yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' },
  legend: { orientation: 'h', y: -0.2 },
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

const state = {
  instances: [],
  results: [],
  currentInstance: null,
  currentInstanceData: null,
  currentResults: [],
  algoFilter: '',
  bucketFilter: '',
  table: null,
};

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
  const [instances, results] = await Promise.all([
    fetchJSON('/api/instances'),
    fetchJSON('/api/results'),
  ]);
  state.instances = instances;
  state.results = results;

  populateKPIs();
  populateControls();
  buildTable();
  drawScatter();

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
    opt.textContent = a;
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
  drawConvergence(results);
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
      name: r.algorithm,
      x: inst.asset_tickers,
      y: normalizeWeights(r.bitstring),
      marker: { color: colorFor(r.algorithm) },
      hovertemplate: `${r.algorithm}<br>%{x}: %{y:.3f}<extra></extra>`,
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

function drawConvergence(results) {
  const withHist = results.filter(
    (r) => Array.isArray(r.convergence_history) && r.convergence_history.length > 1,
  );
  if (!withHist.length) {
    Plotly.purge('chart-convergence');
    document.getElementById('chart-convergence').innerHTML =
      '<p style="color:#8a97ad; padding:24px;">No iterative-optimizer history for this instance.</p>';
    return;
  }
  const traces = withHist.map((r) => {
    const p = r.hyperparameters && r.hyperparameters.p;
    const label = p !== undefined ? `${r.algorithm} (p=${p})` : r.algorithm;
    return {
      type: 'scatter',
      mode: 'lines+markers',
      name: label,
      x: r.convergence_history.map((_, i) => i),
      y: r.convergence_history,
      line: { color: colorFor(r.algorithm), width: 2 },
      marker: { size: 4 },
    };
  });
  const layout = {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'iteration' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'objective' },
  };
  Plotly.newPlot('chart-convergence', traces, layout, PLOTLY_CFG);
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
    name: algo,
    x: items.map((r) => r.wall_time_seconds),
    y: items.map((r) => r.objective_value),
    text: items.map((r) => `${r.instance_id}<br>feasible: ${r.feasible}`),
    hovertemplate: `${algo}<br>%{text}<br>time=%{x:.3f}s<br>obj=%{y:.4f}<extra></extra>`,
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
    r.algorithm,
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

init().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    'afterbegin',
    `<div style="padding:14px; background:#3a1a1a; color:#ffb;">Failed to load: ${err.message}</div>`,
  );
});
