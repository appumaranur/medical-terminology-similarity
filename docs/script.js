// Set this to the public URL of the Flask service on Render.
const API_BASE_URL = 'https://medical-terminology-similarity.onrender.com';

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

function percentage(value) {
  return Number(value || 0).toFixed(1);
}

function apiErrorMessage(response, data) {
  if (response.status === 503) return 'Backend unavailable: the search model is not ready.';
  if (response.status === 400) return data.error || 'Invalid query. Please enter at least two characters.';
  return data.error || `API error (${response.status}). Please try again.`;
}

function renderResults(data) {
  const results = Array.isArray(data.results) ? data.results : [];
  const stats = data.stats || {};
  const comparison = data.comparison || {};

  resultsGrid.innerHTML = results.map((result, index) => {
    const relatedTerms = Array.isArray(result.related_terms) && result.related_terms.length
      ? result.related_terms.join(', ')
      : 'None listed';
    return `
      <article class="result-card">
        <div class="card-top"><span class="rank">0${index + 1}</span><span class="badge">${escapeHtml(result.badge)}</span></div>
        <h3>${escapeHtml(result.term)}</h3>
        <span class="category">${escapeHtml(result.category)}</span>
        <p class="description">${escapeHtml(result.description)}</p>
        <p class="relation"><strong>Relation type:</strong> ${escapeHtml(result.relation)}</p>
        <p class="relation"><strong>Related terms:</strong> ${escapeHtml(relatedTerms)}</p>
        <div class="score-row"><span class="score">${percentage(result.percentage)}%</span><span class="score-label">SIMILARITY</span></div>
        <div class="progress-track"><div class="progress-fill" style="width: ${Math.max(0, Math.min(100, Number(result.percentage) || 0))}%"></div></div>
      </article>
    `;
  }).join('');

  emptyState.hidden = results.length > 0;
  resultQuery.textContent = `Query: ${data.query || ''}`;
  document.getElementById('totalTerms').textContent = stats.total_terms ?? '-';
  document.getElementById('resultsFound').textContent = stats.results_found ?? results.length;
  document.getElementById('bestMatch').textContent = `${percentage(stats.highest_similarity)}%`;
  document.getElementById('averageScore').textContent = `${percentage(stats.average_similarity)}%`;

  comparisonBody.innerHTML = [
    ['Sentence-BERT + FAISS', comparison.sentence_bert],
    ['TF-IDF', comparison.tfidf],
  ].map(([method, result]) => result
    ? `<tr><td><strong>${method}</strong></td><td>${escapeHtml(result.term)}</td><td>${percentage(result.percentage)}%</td></tr>`
    : '').join('') || '<tr><td colspan="3" class="muted-cell">No comparison results returned.</td></tr>';

  similarityChart.innerHTML = results.map((result) => `
    <div class="bar-row"><span class="bar-label" title="${escapeHtml(result.term)}">${escapeHtml(result.term)}</span><div class="bar-track"><div class="bar-fill" style="width: ${Math.max(0, Math.min(100, Number(result.percentage) || 0))}%"></div></div><span class="bar-value">${percentage(result.percentage)}%</span></div>
  `).join('') || '<p class="muted-copy">No semantic results returned.</p>';
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

async function search() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus('Please enter a medical term to search.', true);
    queryInput.focus();
    return;
  }

  searchButton.disabled = true;
  setStatus('Searching...');
  try {
    const response = await fetch(`${API_BASE_URL}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await readJson(response);
    if (!response.ok) throw new Error(apiErrorMessage(response, data));
    renderResults(data);
    setStatus(`Search complete. Showing ${data.stats.results_found} closest terms.`);
  } catch (error) {
    setStatus(error instanceof TypeError ? 'Network error: could not reach the backend.' : error.message, true);
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

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    const data = await readJson(response);
    if (!response.ok) throw new Error(apiErrorMessage(response, data));
    if (data.ready) {
      document.getElementById('totalTerms').textContent = data.total_terms ?? '-';
      setStatus(`Ready. ${data.total_terms} medical terms indexed.`);
    } else {
      setStatus('Backend unavailable: the model is not ready.', true);
    }
  } catch (error) {
    setStatus(error instanceof TypeError ? 'Network error: could not reach the backend.' : error.message, true);
  }
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

checkHealth();
