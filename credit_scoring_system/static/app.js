const state = {
  metadata: null,
};

function optionHtml(options, selected) {
  return Object.entries(options)
    .map(([value, label]) => {
      const isSelected = value === selected ? "selected" : "";
      return `<option value="${value}" ${isSelected}>${label}</option>`;
    })
    .join("");
}

function fieldHtml(field, defaults) {
  const value = defaults[field.name];
  if (field.type === "select") {
    return `
      <label>
        ${field.label}
        <select name="${field.name}" required>
          ${optionHtml(field.options, value)}
        </select>
      </label>
    `;
  }

  return `
    <label>
      ${field.label}
      <input
        type="number"
        name="${field.name}"
        value="${value}"
        min="${field.min}"
        max="${field.max}"
        step="1"
        required
      />
    </label>
  `;
}

function renderForm() {
  const form = document.querySelector("#creditForm");
  form.innerHTML = state.metadata.field_groups
    .map(
      (group) => `
        <section class="group">
          <h2>${group.title}</h2>
          <div class="grid">
            ${group.fields.map((field) => fieldHtml(field, state.metadata.defaults)).join("")}
          </div>
        </section>
      `
    )
    .join("");
}

function renderModels() {
  const select = document.querySelector("#modelSelect");
  select.innerHTML = Object.entries(state.metadata.models)
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
  select.value = "random_forest";
}

function collectApplicant() {
  const formData = new FormData(document.querySelector("#creditForm"));
  const applicant = {};
  for (const [key, value] of formData.entries()) {
    const numericValue = Number(value);
    applicant[key] = Number.isNaN(numericValue) || value.trim().startsWith("A") ? value : numericValue;
  }
  return applicant;
}

function formatPercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function renderResult(result) {
  const panel = document.querySelector(".result-panel");
  const isBad = result.prediction === 1;
  panel.classList.toggle("bad", isBad);
  panel.classList.toggle("good", !isBad);

  document.querySelector("#riskLabel").textContent = isBad ? "High Credit Risk" : "Good Credit Profile";
  document.querySelector("#probabilityText").textContent =
    `Model: ${result.model_name}. Bad-credit probability: ${formatPercent(result.probability_bad)}.`;
  document.querySelector("#meterFill").style.width = formatPercent(result.probability_bad);

  document.querySelector("#decisionBox").innerHTML = isBad
    ? "<strong>Review recommended</strong><span>The model flags this applicant as higher risk. Check income, collateral, repayment history, and loan amount before approval.</span>"
    : "<strong>Likely approvable</strong><span>The model sees a lower-risk profile. Final decisions should still include document verification and policy checks.</span>";
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  state.metadata = await response.json();
  renderModels();
  renderForm();
}

async function submitPrediction(event) {
  event.preventDefault();
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: document.querySelector("#modelSelect").value,
      applicant: collectApplicant(),
    }),
  });
  const result = await response.json();
  if (!response.ok) {
    alert(result.error || "Prediction failed");
    return;
  }
  renderResult(result);
}

document.querySelector("#creditForm").addEventListener("submit", submitPrediction);
document.querySelector("#resetButton").addEventListener("click", renderForm);

loadMetadata();
