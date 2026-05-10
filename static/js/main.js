/* MediAI — main.js */

// ── Symptom checkbox interaction ──────────────────────────────────────────────
function initSymptoms() {
  const counter = document.getElementById('symptom-count');

  function updateCount() {
    const n = document.querySelectorAll('.symptom-item input[type=checkbox]:checked').length;
    if (counter) counter.textContent = n + ' selected';
  }

  // Use event delegation on the parent grid — only listen to checkbox change events
  // (clicking the label natively toggles the checkbox; we just react to the change)
  const grid = document.getElementById('symptom-grid');
  if (grid) {
    grid.addEventListener('change', function(e) {
      if (e.target && e.target.type === 'checkbox') {
        const item = e.target.closest('.symptom-item');
        if (item) item.classList.toggle('active', e.target.checked);
        updateCount();
      }
    });
  }

  // Restore active state on page load (for pre-checked boxes after form POST)
  document.querySelectorAll('.symptom-item input[type=checkbox]:checked').forEach(cb => {
    cb.closest('.symptom-item')?.classList.add('active');
  });

  updateCount();
}

// ── Animated number counters ─────────────────────────────────────────────────
function initCounters() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    if (isNaN(target)) return;
    let start = 0;
    const step = Math.ceil(target / 60);
    const timer = setInterval(() => {
      start = Math.min(start + step, target);
      el.textContent = start.toLocaleString();
      if (start >= target) clearInterval(timer);
    }, 16);
  });
}

// ── Probability bar animations ───────────────────────────────────────────────
function initProbBars() {
  setTimeout(() => {
    document.querySelectorAll('.prob-fill[data-width]').forEach(bar => {
      bar.style.transition = 'width 0.8s cubic-bezier(.4,0,.2,1)';
      bar.style.width = bar.dataset.width + '%';
    });
  }, 200);
}

// ── Plotly chart rendering ───────────────────────────────────────────────────
function renderCharts() {
  if (typeof Plotly === 'undefined') {
    console.error('Plotly is not loaded!');
    return;
  }
  
  console.log('Plotly version:', Plotly.version);
  console.log('Found chart elements:', document.querySelectorAll('[data-chart]').length);
  
  document.querySelectorAll('[data-chart]').forEach((el, index) => {
    try {
      const chartData = el.dataset.chart;
      console.log(`Chart ${index}: data-chart attribute length:`, chartData ? chartData.length : 0);
      
      if (!chartData || chartData === '{}' || chartData === 'null') {
        console.warn(`Chart ${index}: Empty or null chart data`);
        el.innerHTML = '<div style="padding:20px;text-align:center;color:#94a3b8">No data available</div>';
        return;
      }
      
      const data = JSON.parse(chartData);
      console.log(`Chart ${index}: Parsed data:`, {
        hasData: !!data.data,
        hasLayout: !!data.layout,
        traces: data.data ? data.data.length : 0
      });
      
      if (!data.data || data.data.length === 0) {
        console.warn(`Chart ${index}: No data traces`);
        el.innerHTML = '<div style="padding:20px;text-align:center;color:#94a3b8">No data traces</div>';
        return;
      }
      
      // Plotly.newPlot automatically handles binary data format
      Plotly.newPlot(el, data.data, data.layout, { 
        responsive: true, 
        displayModeBar: false 
      }).then(() => {
        console.log(`Chart ${index}: Rendered successfully`);
      }).catch(err => {
        console.error(`Chart ${index}: Plotly render error:`, err);
        el.innerHTML = '<div style="padding:20px;text-align:center;color:#ff4d6d">Error rendering chart</div>';
      });
      
    } catch (e) {
      console.error(`Chart ${index}: Exception:`, e, 'Element:', el);
      el.innerHTML = '<div style="padding:20px;text-align:center;color:#ff4d6d">Error: ' + e.message + '</div>';
    }
  });
}

// ── Nav dropdown ─────────────────────────────────────────────────────────────
function initDropdowns() {
  document.querySelectorAll('.nav-dropdown').forEach(dd => {
    dd.addEventListener('mouseenter', () => dd.classList.add('open'));
    dd.addEventListener('mouseleave', () => dd.classList.remove('open'));
  });
}

// ── Clear all button ─────────────────────────────────────────────────────────
function initClearAll() {
  const btn = document.getElementById('clear-all');
  if (!btn) return;
  btn.addEventListener('click', () => {
    document.querySelectorAll('.symptom-item input[type=checkbox]').forEach(cb => {
      cb.checked = false;
      cb.closest('.symptom-item')?.classList.remove('active');
    });
    const counter = document.getElementById('symptom-count');
    if (counter) counter.textContent = '0 selected';
  });
}

// ── Symptom search filter ─────────────────────────────────────────────────────
function initSearch() {
  const searchInput = document.getElementById('symptom-search');
  if (!searchInput) return;
  searchInput.addEventListener('input', function() {
    const q = this.value.toLowerCase();
    document.querySelectorAll('.symptom-item').forEach(item => {
      const label = (item.dataset.label || '').toLowerCase();
      item.style.display = label.includes(q) ? '' : 'none';
    });
  });
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initSymptoms();
  initCounters();
  initProbBars();
  renderCharts();
  initDropdowns();
  initClearAll();
  initSearch();
});
