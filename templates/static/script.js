const form = document.getElementById("visaForm");
const submitBtn = document.getElementById("submitBtn");
const resultEl = document.getElementById("result");

form.addEventListener("submit", async function (e) {
  e.preventDefault();

  if (!validateForm()) {
    return;
  }

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

  const approvedPct = round1(result.probabilities.approved);
  const rejectedPct = round1(result.probabilities.rejected);
  const confidence = round1(result.confidence);

  const statusClass = approved ? "approved" : "denied";
  const statusText = approved ? "Approved" : "Denied";
  const ringColor = approved ? "#10b981" : "#ef4444";

  const factors = result.top_features || [];

  // Robust Insight Generator
  const insights = generateInsightsAndRecommendations(factors);

  const factorsHtml = factors.length
    ? factors.map(renderFactorCard).join("")
    : `<div class="no-factors">Feature contribution data is unavailable for this prediction.</div>`;

  resultEl.hidden = false;
  resultEl.classList.remove("is-error");

  resultEl.innerHTML = `
    <!-- Main Outcome Display -->
    <div class="outcome-banner ${statusClass}">
      <div>
        <span class="outcome-tag">AI Assessment Result</span>
        <h2 class="outcome-title">${escapeHtml(result.prediction)}</h2>
        <p class="outcome-subtitle">Confidence ${confidence}% • Generated using trained ML model</p>
      </div>
      <span class="outcome-badge ${statusClass}">${statusText}</span>
    </div>

    <!-- Probability & Metric Visualizations -->
    <div class="metrics-row">
      <div class="confidence-ring-card">
        <span class="metric-heading">Overall Confidence</span>
        <div class="circle-chart">
          <svg width="120" height="120" viewBox="0 0 36 36">
            <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
            <path class="circle-fill" stroke="${ringColor}" data-target="${confidence},100" stroke-dasharray="0,100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
          </svg>
          <div class="circle-text"><strong>${confidence}%</strong></div>
        </div>
      </div>

      <div class="distribution-card">
        <div class="distribution-header">
          <span>Prediction Probability</span>
          <span>Total = 100%</span>
        </div>
        <div class="split-progress-bar">
          <div class="seg-approved" style="width:0%" data-target="${approvedPct}%"></div>
          <div class="seg-rejected" style="width:0%" data-target="${rejectedPct}%"></div>
        </div>
        <div class="legend-row">
          <span class="legend-approved">● Approval ${approvedPct}%</span>
          <span class="legend-denied">● Rejection ${rejectedPct}%</span>
        </div>
      </div>
    </div>

    <!-- AI Natural Language Reasoning & Recommendations Cards -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.25rem; margin-top: 1rem;">
      
      <!-- Key Factors Card -->
      <div style="background: ${approved ? 'rgba(236,253,245,0.8)' : 'rgba(254,242,242,0.8)'}; border: 1px solid ${approved ? '#a7f3d0' : '#fecdd3'}; border-radius: 16px; padding: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem; color: ${approved ? '#047857' : '#b91c1c'}; font-weight: 800; font-size: 0.95rem;">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>${approved ? 'Key Factors Supporting Approval' : 'Primary Risk & Rejection Drivers'}</span>
        </div>
        <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.65rem; font-size: 0.875rem; color: #0f172a;">
          ${insights.reasons.map(reason => `<li style="position: relative; padding-left: 1.25rem;">• ${reason}</li>`).join('')}
        </ul>
      </div>

      <!-- Actionable Steps Card -->
      <div style="background: rgba(238,242,255,0.8); border: 1px solid #c7d2fe; border-radius: 16px; padding: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem; color: #4338ca; font-weight: 800; font-size: 0.95rem;">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
          <span>Actionable Steps for Improvement</span>
        </div>
        <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.65rem; font-size: 0.875rem; color: #0f172a;">
          ${insights.recommendations.map(rec => `<li style="background: rgba(255,255,255,0.7); padding: 0.5rem 0.75rem; border-radius: 8px; border: 1px solid rgba(199,210,254,0.5);">💡 ${rec}</li>`).join('')}
        </ul>
      </div>

    </div>

    <!-- SHAP Weight Breakdown Section -->
    <div class="factors-section" style="margin-top: 1rem;">
      <h3 class="factors-title">Top Factors Influencing This Decision</h3>
      <div class="factors-grid">${factorsHtml}</div>
    </div>
  `;

  animateInResultVisuals(resultEl);
  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFactorCard(factor) {
  const impact = Number(factor.impact) || 0;
  const positive = impact > 0;
  const percentage = Math.min(Math.abs(impact) / 3 * 100, 100);
  const featureName = prettifyFeatureName(factor.feature);
  const direction = positive ? "Supports Approval" : "Supports Denial";

  // Dynamic background & border based on impact direction
  const cardBg = positive ? "rgba(236, 253, 245, 0.75)" : "rgba(254, 242, 242, 0.75)";
  const cardBorder = positive ? "#a7f3d0" : "#fecdd3";

  return `
    <div class="factor-card ${positive ? 'factor-pos' : 'factor-neg'}" 
         style="background: ${cardBg}; border: 1.5px solid ${cardBorder}; border-radius: var(--radius-md); padding: 1rem; display: flex; flex-direction: column; gap: 0.5rem; transition: transform 0.2s ease, box-shadow 0.2s ease;">
      <div class="factor-top">
        <div>
          <div class="factor-feature-name" style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary);">
            ${featureName}
          </div>
          <div class="factor-direction ${positive ? "pos" : "neg"}">
            ${direction}
          </div>
        </div>
        <span class="factor-score-pill ${positive ? "pos" : "neg"}">
          ${impact > 0 ? "+" : ""}${impact.toFixed(2)}
        </span>
      </div>
      <div class="factor-bar-track">
        <div class="factor-bar-inner ${positive ? "pos" : "neg"}" data-target="${percentage}%" style="width:0%"></div>
      </div>
    </div>
  `;
}

// FULLY DYNAMIC FEATURE IMPACT PARSER
function generateInsightsAndRecommendations(factors) {
  const reasons = [];
  const recommendations = [];

  factors.forEach(f => {
    const rawName = String(f.feature || "").toLowerCase();
    const impact = Number(f.impact) || 0;
    const readableName = prettifyFeatureName(f.feature);

    if (impact < 0) {
      // Negative impacts -> Actionable Recommendations & Risk Points
      if (rawName.includes("wage") || rawName.includes("salary")) {
        recommendations.push("<strong>Wage Adjustment:</strong> Raise base salary offer closer to regional industry standards.");
      } else if (rawName.includes("edu")) {
        recommendations.push("<strong>Education Proof:</strong> Include documentation for advanced degrees or technical credentials.");
      } else if (rawName.includes("exp") || rawName.includes("work")) {
        recommendations.push("<strong>Experience Verification:</strong> Attach official past employment verification letters or lead portfolios.");
      } else if (rawName.includes("age") || rawName.includes("estab")) {
        recommendations.push("<strong>Sponsor Documentation:</strong> Submit corporate financial tax statements or audit reports.");
      } else {
        recommendations.push(`<strong>${readableName}:</strong> Review provided details for compliance and updates.`);
      }
    } else if (impact > 0) {
      // Positive impacts -> Highlight in Supporting Approval Card
      if (rawName.includes("location") || rawName.includes("region")) {
        reasons.push("Selected employment region strongly supports local labor demand metrics.");
      } else if (rawName.includes("size") || rawName.includes("employee")) {
        reasons.push("Employer workforce scale provides high organizational stability confidence.");
      } else {
        reasons.push(`${readableName} acts as a strong positive driver for approval.`);
      }
    }
  });

  if (reasons.length === 0) {
    reasons.push("Petition parameters align within acceptable standard evaluation thresholds.");
  }
  if (recommendations.length === 0) {
    recommendations.push("<strong>Application Review:</strong> Double-check all official sponsorship filing forms for accuracy.");
  }

  return { reasons, recommendations };
}

function animateInResultVisuals(root) {
  requestAnimationFrame(() => {
    const ring = root.querySelector(".circle-fill");
    if (ring) {
      ring.setAttribute("stroke-dasharray", ring.dataset.target);
    }

    root.querySelectorAll(".seg-approved,.seg-rejected,.factor-bar-inner").forEach(el => {
      el.style.width = el.dataset.target;
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

function validateForm() {
  const fields = [
    { element: document.getElementById("employees"), message: "Please enter company headcount." },
    { element: document.getElementById("year"), message: "Please enter establishment year." },
    { element: document.getElementById("wage"), message: "Please enter prevailing wage." }
  ];

  for (const item of fields) {
    item.element.classList.remove("input-error");

    if (item.element.value.trim() === "") {
      item.element.classList.add("input-error");
      item.element.focus();
      showToast(item.message);
      return false;
    }
  }

  const year = Number(document.getElementById("year").value);
  if (year < 1800 || year > new Date().getFullYear()) {
    document.getElementById("year").classList.add("input-error");
    showToast("Please enter a valid establishment year.");
    document.getElementById("year").focus();
    return false;
  }

  const wage = Number(document.getElementById("wage").value);
  if (wage <= 0) {
    document.getElementById("wage").classList.add("input-error");
    showToast("Prevailing wage must be greater than zero.");
    document.getElementById("wage").focus();
    return false;
  }

  return true;
}

function showToast(message) {
  let toast = document.querySelector(".toast");

  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }

  toast.textContent = message;
  toast.classList.add("show");

  setTimeout(() => {
    toast.classList.remove("show");
  }, 2500);
}

document.querySelectorAll("input,select").forEach(el => {
  el.addEventListener("input", () => el.classList.remove("input-error"));
  el.addEventListener("change", () => el.classList.remove("input-error"));
});

function prettifyFeatureName(name) {
  if (!name) return "";
  name = name
    .replace(/^Transformer__/, "")
    .replace(/^StandardScaler__/, "")
    .replace(/^Ordinal_Encoder__/, "")
    .replace(/^OrdinalEncoder__/, "");

  const mapping = {
    company_age: "Company Age",
    employer_age: "Employer Age",
    no_of_employees: "Company Size",
    employer_size: "Employer Size",
    prevailing_wage: "Prevailing Wage",
    education_of_employee: "Education Level",
    education_level: "Education Level",
    continent: "Applicant Origin",
    region_of_employment: "Employment Region",
    job_location: "Job Location",
    unit_of_wage: "Salary Frequency",
    has_job_experience: "Prior Work Experience",
    previous_work_experience: "Previous Work Experience",
    requires_job_training: "Job Training Required",
    full_time_position: "Full-Time Position"
  };

  return mapping[name] || name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}