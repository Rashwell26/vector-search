const searchForm = document.querySelector('#search-form');
const queryInput = document.querySelector('#query-input');
const kSelect = document.querySelector('#k-select');
const searchButton = document.querySelector('#search-button');
const resultsList = document.querySelector('#results');
const resultsMeta = document.querySelector('#results-meta');
const latencyLabel = document.querySelector('#latency');
const statusBadge = document.querySelector('#status-badge');
const examplesList = document.querySelector('#examples');

const EXAMPLE_QUERIES = [
  'how do database indexes work',
  'what causes a deadlock',
  'make computers agree despite faulty nodes',
  'sharing memory between threads',
];

function el(tag, className = '', text = '') {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function setStatus(message, state = '') {
  statusBadge.textContent = message;
  statusBadge.className = `status-badge${state === '' ? '' : ` is-${state}`}`;
}

async function loadHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    setStatus(`${data.documents} documents indexed`, 'online');
  } catch {
    setStatus('API offline', 'offline');
  }
}

function createResultCard(result, rank) {
  const card = el('li', 'result-card');

  const header = el('div', 'result-header');
  header.append(
    el('span', 'result-rank', `#${rank}`),
    el('span', 'result-source', result.source),
    el('span', 'result-score', result.score.toFixed(3)),
  );

  const barTrack = el('div', 'score-track');
  const barFill = el('div', 'score-fill');
  barFill.style.width = `${Math.round(Math.max(result.score, 0) * 100)}%`;
  barTrack.append(barFill);

  card.append(header, barTrack, el('p', 'result-text', result.text));
  return card;
}

function showMessage(className, text) {
  resultsList.replaceChildren(el('li', className, text));
}

function renderResults(results, elapsedMs) {
  const cards = results.map((result, index) => createResultCard(result, index + 1));
  resultsList.replaceChildren(...cards);
  resultsMeta.hidden = false;
  latencyLabel.textContent = `${results.length} results in ${elapsedMs} ms`;
}

async function runSearch(query, k) {
  searchButton.disabled = true;
  showMessage('empty-state', 'Searching…');
  resultsMeta.hidden = true;

  const startedAt = performance.now();
  try {
    const params = new URLSearchParams({ q: query, k: String(k) });
    const response = await fetch(`/search?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const elapsedMs = Math.round(performance.now() - startedAt);

    if (data.results.length === 0) {
      showMessage('empty-state', 'No matching documents found.');
      return;
    }
    renderResults(data.results, elapsedMs);
  } catch (error) {
    showMessage('error-state', `Search failed: ${error.message}`);
  } finally {
    searchButton.disabled = false;
  }
}

searchForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (query === '') {
    return;
  }
  runSearch(query, Number(kSelect.value));
});

EXAMPLE_QUERIES.forEach((example) => {
  const chip = el('button', 'example-chip', example);
  chip.type = 'button';
  chip.addEventListener('click', () => {
    queryInput.value = example;
    runSearch(example, Number(kSelect.value));
  });
  examplesList.append(chip);
});

loadHealth();
