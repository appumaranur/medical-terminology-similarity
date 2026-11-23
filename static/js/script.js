/* ── DOM refs ── */
const queryInput     = document.getElementById('queryInput');
const searchButton   = document.getElementById('searchButton');
const clearButton    = document.getElementById('clearButton');
const statusMessage  = document.getElementById('statusMessage');
const resultsGrid    = document.getElementById('resultsGrid');
const emptyState     = document.getElementById('emptyState');
const resultQuery    = document.getElementById('resultQuery');
const comparisonBody = document.getElementById('comparisonBody');
const similarityChart= document.getElementById('similarityChart');
const exportRow      = document.getElementById('exportRow');
const suggestList    = document.getElementById('suggestList');
const batchInput     = document.getElementById('batchInput');
const batchButton    = document.getElementById('batchButton');
const batchStatus    = document.getElementById('batchStatus');
const batchResults   = document.getElementById('batchResults');
const themeToggle    = document.getElementById('themeToggle');

let lastQuery = '';
let suggestIndex = -1;

/* ── Helpers ── */
function escapeHtml(value) {
  const el = document.createElement('div');
  el.textContent = value || '';
  return el.innerHTML;
}

function setStatus(msg, type = '') {
  statusMessage.textContent = msg;
  statusMessage.className = 'status-message' + (type ? ` ${type}-message` : '');
}

function spinner() {
  return '<span class="spinner" aria-hidden="true"></span>';
}

/* ── Dark mode (persisted) ── */
(function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.dataset.theme = saved;
  themeToggle.textContent = saved === 'dark' ? 'Light mode' : 'Dark mode';
})();

themeToggle.addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
  themeToggle.textContent = dark ? 'Dark mode' : 'Light mode';
});

/* ── Autocomplete ── */
let suggestTimer;
queryInput.addEventListener('input', () => {
  clearTimeout(suggestTimer);
  const q = queryInput.value.trim();
  if (q.length < 2) { hideSuggest(); return; }
  suggestTimer = setTimeout(() => fetchSuggest(q), 220);
});

async function fetchSuggest(q) {
  try {
    const res = await fetch(`/api/suggest?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    renderSuggest(data.suggestions || []);
  } catch (_) {}
}

function renderSuggest(items) {
  if (!items.length) { hideSuggest(); return; }
  suggestList.innerHTML = items.map((s, i) =>
    `<li role="option" data-idx="${i}" tabindex="-1">${escapeHtml(s)}</li>`
  ).join('');
  suggestList.hidden = false;
  suggestIndex = -1;
  suggestList.querySelectorAll('li').forEach(li => {
    li.addEventListener('mousedown', e => {
      e.preventDefault();
      queryInput.value = li.textContent;
      hideSuggest();
      search();
    });
  });
}

function hideSuggest() { suggestList.hidden = true; suggestList.innerHTML = ''; suggestIndex = -1; }

queryInput.addEventListener('keydown', e => {
  if (suggestList.hidden) {
    if (e.key === 'Enter') search();
    return;
  }
  const items = suggestList.querySelectorAll('li');
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    suggestIndex = Math.min(suggestIndex + 1, items.length - 1);
    updateSuggestActive(items);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    suggestIndex = Math.max(suggestIndex - 1, -1);
    updateSuggestActive(items);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (suggestIndex >= 0 && items[suggestIndex]) {
      queryInput.value = items[suggestIndex].textContent;
      hideSuggest();
    }
    search();
  } else if (e.key === 'Escape') {
    hideSuggest();
  }
});

function updateSuggestActive(items) {
  items.forEach((li, i) => {
    li.setAttribute('aria-selected', i === suggestIndex ? 'true' : 'false');
    if (i === suggestIndex) { li.scrollIntoView({ block: 'nearest' }); queryInput.value = li.textContent; }
  });
}

document.addEventListener('click', e => {
  if (!suggestList.contains(e.target) && e.target !== queryInput) hideSuggest();
});

/* ── Render results ── */
function renderResults(data) {
  resultsGrid.innerHTML = data.results.map((result, index) => `
    <article class="result-card">
      <div class="card-top"><span class="rank">0${index + 1}</span><span class="badge">${escapeHtml(result.badge)}</span></div>
      <h3>${escapeHtml(result.term)}</h3>
      <span class="category">${escapeHtml(result.category)}</span>
      <p class="description">${escapeHtml(result.description)}</p>
      ${result.related_terms && result.related_terms.length ? `
        <div class="related-tags">
          ${result.related_terms.slice(0, 4).map(t => `<button class="related-tag" data-query="${escapeHtml(t)}">${escapeHtml(t)}</button>`).join('')}
        </div>` : ''}
      <p class="relation"><strong>Relation:</strong> ${escapeHtml(result.relation)}</p>
      <div class="score-row"><span class="score">${result.percentage.toFixed(1)}%</span><span class="score-label">SIMILARITY</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:${result.percentage}%"></div></div>
    </article>
  `).join('');

  // Related tag click → new search
  resultsGrid.querySelectorAll('.related-tag').forEach(btn => {
    btn.addEventListener('click', () => { queryInput.value = btn.dataset.query; search(); });
  });

  emptyState.hidden = true;
  resultQuery.textContent = `Query: ${data.query}`;
  document.getElementById('totalTerms').textContent = data.stats.total_terms;
  document.getElementById('resultsFound').textContent = data.stats.results_found;
  document.getElementById('bestMatch').textContent = `${data.stats.highest_similarity.toFixed(1)}%`;
  document.getElementById('averageScore').textContent = `${data.stats.average_similarity.toFixed(1)}%`;

  comparisonBody.innerHTML = [
    ['Sentence-BERT', data.comparison.sentence_bert],
    ['TF-IDF', data.comparison.tfidf],
  ].map(([method, result]) => result
    ? `<tr><td><strong>${method}</strong></td><td>${escapeHtml(result.term)}</td><td>${result.percentage.toFixed(1)}%</td></tr>`
    : ''
  ).join('');

  similarityChart.innerHTML = data.results.map(result => `
    <div class="bar-row">
      <span class="bar-label" title="${escapeHtml(result.term)}">${escapeHtml(result.term)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${result.percentage}%"></div></div>
      <span class="bar-value">${result.percentage.toFixed(1)}%</span>
    </div>
  `).join('');

  exportRow.hidden = false;
}

/* ── Single search ── */
async function search() {
  const query = queryInput.value.trim();
  hideSuggest();
  if (!query) { setStatus('Please enter a medical term to search.', 'error'); queryInput.focus(); return; }
  lastQuery = query;
  searchButton.disabled = true;
  exportRow.hidden = true;
  setStatus(`${spinner()}Searching semantic embeddings...`);
  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Search failed.');
    renderResults(data);
    setStatus(`Search complete — ${data.stats.results_found} closest terms found.`, 'ready');
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    searchButton.disabled = false;
  }
}

/* ── Clear ── */
function clearSearch() {
  queryInput.value = '';
  lastQuery = '';
  resultsGrid.innerHTML = '';
  emptyState.hidden = false;
  resultQuery.textContent = 'Awaiting a search';
  comparisonBody.innerHTML = '<tr><td colspan="3" class="muted-cell">Run a search to compare methods.</td></tr>';
  similarityChart.innerHTML = '<p class="muted-copy">Run a search to see the score distribution.</p>';
  document.getElementById('resultsFound').textContent = '-';
  document.getElementById('bestMatch').textContent = '-';
  document.getElementById('averageScore').textContent = '-';
  exportRow.hidden = true;
  setStatus('Ready for another search.');
  queryInput.focus();
}

/* ── Export ── */
document.querySelectorAll('.export-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!lastQuery) return;
    const fmt = btn.dataset.fmt;
    try {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: lastQuery, format: fmt }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.error); }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `medmatch_${lastQuery.slice(0, 30).replace(/\s+/g, '_')}.${fmt}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setStatus(`Export failed: ${err.message}`, 'error');
    }
  });
});

/* ── Batch search ── */
batchButton.addEventListener('click', async () => {
  const lines = batchInput.value.split('\n').map(l => l.trim()).filter(l => l.length >= 2).slice(0, 20);
  if (!lines.length) { batchStatus.textContent = 'Enter at least one term.'; return; }
  batchButton.disabled = true;
  batchStatus.innerHTML = `${spinner()}Running batch search for ${lines.length} terms…`;
  batchResults.innerHTML = '';
  try {
    const res = await fetch('/api/batch_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ queries: lines }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Batch search failed.');
    renderBatchResults(data.batch_results);
    batchStatus.textContent = `Done — ${data.count} queries processed.`;
  } catch (err) {
    batchStatus.textContent = `Error: ${err.message}`;
  } finally {
    batchButton.disabled = false;
  }
});

function renderBatchResults(batchResults_data) {
  batchResults.innerHTML = batchResults_data.map(item => `
    <div class="batch-item">
      <div class="batch-item-header">
        <span class="batch-query-label">${escapeHtml(item.query)}</span>
        <span class="batch-top-match">${item.results[0] ? `Best: ${escapeHtml(item.results[0].term)} (${item.results[0].percentage.toFixed(1)}%)` : 'No results'}</span>
      </div>
      <div class="batch-mini-grid">
        ${item.results.map((r, i) => `
          <div class="batch-mini-card">
            <strong>${escapeHtml(r.term)}</strong>
            <span class="batch-mini-score">${r.percentage.toFixed(1)}%</span>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');
}

/* ── Event wiring ── */
searchButton.addEventListener('click', search);
clearButton.addEventListener('click', clearSearch);
document.querySelectorAll('.example-button').forEach(btn => {
  btn.addEventListener('click', () => { queryInput.value = btn.dataset.query; search(); });
});

/* ── Health check on load ── */
fetch('/api/health')
  .then(r => r.json())
  .then(data => {
    if (data.ready) setStatus(`Ready — ${data.total_terms} medical terms indexed.`, 'ready');
    else setStatus('The model is not ready. Restart the app after checking dependencies.', 'error');
  })
  .catch(() => setStatus('Could not connect to the backend.', 'error'));
