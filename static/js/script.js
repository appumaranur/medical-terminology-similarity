const queryInput = document.getElementById('queryInput');
const searchButton = document.getElementById('searchButton');
const clearButton = document.getElementById('clearButton');
const statusMessage = document.getElementById('statusMessage');
const resultsGrid = document.getElementById('resultsGrid');
const emptyState = document.getElementById('emptyState');
const resultQuery = document.getElementById('resultQuery');
const comparisonBody = document.getElementById('comparisonBody');
const similarityChart = document.getElementById('similarityChart');

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle('error-message', isError);
}

function escapeHtml(value) {
  const element = document.createElement('div');
  element.textContent = value || '';
  return element.innerHTML;
}

function renderResults(data) {
  resultsGrid.innerHTML = data.results.map((result, index) => `
    <article class="result-card">
      <div class="card-top"><span class="rank">0${index + 1}</span><span class="badge">${escapeHtml(result.badge)}</span></div>
      <h3>${escapeHtml(result.term)}</h3>
      <span class="category">${escapeHtml(result.category)}</span>
      <p class="description">${escapeHtml(result.description)}</p>
      <p class="relation"><strong>Relation:</strong> ${escapeHtml(result.relation)}</p>
      <div class="score-row"><span class="score">${result.percentage.toFixed(1)}%</span><span class="score-label">SIMILARITY</span></div>
      <div class="progress-track"><div class="progress-fill" style="width: ${result.percentage}%"></div></div>
    </article>
  `).join('');
  emptyState.hidden = true;
  resultQuery.textContent = `Query: ${data.query}`;
  document.getElementById('totalTerms').textContent = data.stats.total_terms;
  document.getElementById('resultsFound').textContent = data.stats.results_found;
  document.getElementById('bestMatch').textContent = `${data.stats.highest_similarity.toFixed(1)}%`;
  document.getElementById('averageScore').textContent = `${data.stats.average_similarity.toFixed(1)}%`;

  comparisonBody.innerHTML = [
    ['Sentence-BERT', data.comparison.sentence_bert],
    ['TF-IDF', data.comparison.tfidf],
  ].map(([method, result]) => result ? `<tr><td><strong>${method}</strong></td><td>${escapeHtml(result.term)}</td><td>${result.percentage.toFixed(1)}%</td></tr>` : '').join('');

  similarityChart.innerHTML = data.results.map(result => `
    <div class="bar-row"><span class="bar-label" title="${escapeHtml(result.term)}">${escapeHtml(result.term)}</span><div class="bar-track"><div class="bar-fill" style="width: ${result.percentage}%"></div></div><span class="bar-value">${result.percentage.toFixed(1)}%</span></div>
  `).join('');
}

async function search() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus('Please enter a medical term to search.', true);
    queryInput.focus();
    return;
  }
  searchButton.disabled = true;
  setStatus('Searching semantic embeddings...');
  try {
    const response = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Search failed.');
    renderResults(data);
    setStatus(`Search complete. Showing ${data.stats.results_found} closest terms.`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    searchButton.disabled = false;
  }
}

function clearSearch() {
  queryInput.value = '';
  resultsGrid.innerHTML = '';
  emptyState.hidden = false;
  resultQuery.textContent = 'Awaiting a search';
  comparisonBody.innerHTML = '<tr><td colspan="3" class="muted-cell">Run a search to compare methods.</td></tr>';
  similarityChart.innerHTML = '<p class="muted-copy">Run a search to see the score distribution.</p>';
  document.getElementById('resultsFound').textContent = '-';
  document.getElementById('bestMatch').textContent = '-';
  document.getElementById('averageScore').textContent = '-';
  setStatus('Ready for another search.');
  queryInput.focus();
}

searchButton.addEventListener('click', search);
clearButton.addEventListener('click', clearSearch);
queryInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') search(); });
document.querySelectorAll('.example-button').forEach((button) => button.addEventListener('click', () => { queryInput.value = button.dataset.query; search(); }));
document.getElementById('themeToggle').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  document.getElementById('themeToggle').textContent = dark ? 'Dark mode' : 'Light mode';
});

fetch('/api/health').then((response) => response.json()).then((data) => {
  if (data.ready) setStatus(`Ready. ${data.total_terms} medical terms indexed.`);
  else setStatus('The model is not ready. Restart the app after checking dependencies.', true);
}).catch(() => setStatus('Could not connect to the backend.', true));
