const form = document.getElementById("visaForm");
const submitBtn = document.getElementById("submitBtn");
const resultEl = document.getElementById("result");

form.addEventListener("submit", async function (e) {
  e.preventDefault();

  const getRadioVal = (name) => document.querySelector(`input[name="${name}"]:checked`)?.value;

  const data = {
    continent: document.getElementById("continent").value,
    education_of_employee: document.getElementById("education").value,
    has_job_experience: getRadioVal("experience"),
    requires_job_training: getRadioVal("training"),
    no_of_employees: Number(document.getElementById("employees").value),
    yr_of_estab: Number(document.getElementById("year").value),
    region_of_employment: document.getElementById("region").value,
    prevailing_wage: Number(document.getElementById("wage").value),
    unit_of_wage: document.getElementById("unit").value,
    full_time_position: getRadioVal("fulltime")
  };

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error(`Server responded with code ${response.status}`);
    }

    const result = await response.json();
    renderResult(result);
  } catch (error) {
    renderError(error);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle("is-loading", isLoading);
  const btnText = submitBtn.querySelector(".btn-text");
  if (btnText) {
    btnText.textContent = isLoading ? "Analyzing Case Parameters..." : "Predict Visa Approval";
  }
}

function renderResult(result) {
  const approved = result.prediction === "Visa Approved";
  const approvedPct = round1(result.probabilities?.approved ?? result.confidence);
  const rejectedPct = round1(result.probabilities?.rejected ?? (100 - approvedPct));
  const confidence = round1(result.confidence);

  const statusClass = approved ? "approved" : "denied";
  const statusText = approved ? "Approved" : "Denied";
  const ringColor = approved ? "#10b981" : "#f43f5e";

  const factorsHtml = (result.top_features || []).map(renderFactorCard).join("");

  resultEl.hidden = false;
  resultEl.classList.remove("is-error");

  // NOTE: every data visual below starts at its zero state (dasharray "0,100",
  // width 0%). The CSS already defines transitions for these properties, but a
  // transition only plays on a *change* — setting the final value directly in
  // this innerHTML write was why the ring/bars used to just snap into place.
  resultEl.innerHTML = `
    <!-- Result Header Banner -->
    <div class="outcome-banner ${statusClass}">
      <div>
        <span class="outcome-tag">Prediction Assessment Result</span>
        <h2 class="outcome-title">${escapeHtml(result.prediction)}</h2>
      </div>
      <span class="outcome-badge ${statusClass}">${statusText}</span>
    </div>

    <!-- Gauge & Metrics Grid -->
    <div class="metrics-row">
      <!-- Circular Ring Chart -->
      <div class="confidence-ring-card">
        <strong style="font-size:0.85rem; color:var(--text-primary);">Confidence Rating</strong>
        <div class="circle-chart">
          <svg width="120" height="120" viewBox="0 0 36 36">
            <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
            <path class="circle-fill" data-target="${confidence}, 100" stroke="${ringColor}" stroke-dasharray="0, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
          </svg>
          <div class="circle-text">${confidence}%</div>
        </div>
        <span style="font-size:0.75rem; color:var(--text-muted);">Model Certainty</span>
      </div>

      <!-- Probability Distribution Bar -->
      <div class="distribution-card">
        <div class="distribution-header">
          <span>Probability Breakdown</span>
          <span>100% Total Scale</span>
        </div>
        <div class="split-progress-bar">
          <div class="seg-approved" data-target="${approvedPct}%" style="width:0%"></div>
          <div class="seg-rejected" data-target="${rejectedPct}%" style="width:0%"></div>
        </div>
        <div class="legend-row">
          <span class="legend-approved">&#9679; Approval Probability: ${approvedPct}%</span>
          <span class="legend-denied">&#9679; Denial Probability: ${rejectedPct}%</span>
        </div>
      </div>
    </div>

    <!-- Key Decision Influencers Grid -->
    ${factorsHtml ? `
    <div class="factors-section">
      <h3 class="factors-title">Key Decision Influencers (Feature Contribution)</h3>
      <div class="factors-grid">
        ${factorsHtml}
      </div>
    </div>` : ""}
  `;

  animateInResultVisuals(resultEl);
  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFactorCard(factor) {
  const impact = Number(factor.impact) || 0;
  const isPos = impact >= 0;
  const magnitude = Math.min(Math.abs(impact), 1) * 100;

  return `
    <div class="factor-card">
      <div class="factor-top">
        <span class="factor-feature-name" title="${escapeHtml(factor.feature)}">${escapeHtml(factor.feature)}</span>
        <span class="factor-score-pill ${isPos ? "pos" : "neg"}">
          ${impact > 0 ? "+" : ""}${impact.toFixed(2)}
        </span>
      </div>
      <div class="factor-bar-track">
        <div class="factor-bar-inner ${isPos ? "pos" : "neg"}" data-target="${magnitude}%" style="width:0%"></div>
      </div>
    </div>
  `;
}

// Elements are rendered at their zero state with the real value stashed in
// data-target. Reading it back one frame later gives the browser a genuine
// from -> to change to animate, instead of appearing instantly.
function animateInResultVisuals(root) {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const ring = root.querySelector(".circle-fill");
      if (ring) ring.setAttribute("stroke-dasharray", ring.dataset.target);

      root.querySelectorAll(".seg-approved, .seg-rejected, .factor-bar-inner").forEach((el) => {
        el.style.width = el.dataset.target;
      });
    });
  });
}

function renderError(error) {
  resultEl.hidden = false;
  resultEl.classList.add("is-error");
  resultEl.innerHTML = `
    <div class="error-box">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <circle cx="12" cy="16" r="0.5" fill="currentColor"/>
      </svg>
      <div>
        <h3>Assessment Request Failed</h3>
        <p>${escapeHtml(error.message || String(error))} — check backend ML model service status.</p>
      </div>
    </div>
  `;
  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function round1(n) {
  return Math.round((Number(n) || 0) * 10) / 10;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}