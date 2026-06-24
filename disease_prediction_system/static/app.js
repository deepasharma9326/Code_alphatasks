const state = {
  metadata: {},
  currentModel: "heart",
};

const labels = {
  age: "Age",
  sex: "Sex (1 male, 0 female)",
  chest_pain_type: "Chest pain type (1-4)",
  resting_bp: "Resting blood pressure",
  cholesterol: "Cholesterol",
  fasting_blood_sugar: "Fasting blood sugar > 120 (1 yes, 0 no)",
  rest_ecg: "Resting ECG (0-2)",
  max_heart_rate: "Maximum heart rate",
  exercise_angina: "Exercise angina (1 yes, 0 no)",
  oldpeak: "Oldpeak",
  slope: "Slope (1-3)",
  major_vessels: "Major vessels (0-3)",
  thal: "Thal (3 normal, 6 fixed, 7 reversible)",
  avg_glucose: "Average glucose",
  max_glucose: "Maximum glucose",
  min_glucose: "Minimum glucose",
  glucose_readings: "Number of glucose readings",
  regular_insulin: "Regular insulin total",
  nph_insulin: "NPH insulin total",
  ultralente_insulin: "UltraLente insulin total",
  meal_events: "Meal events",
  exercise_events: "Exercise events",
  hypoglycemic_symptoms: "Hypoglycemic symptoms",
};

function titleFor(feature) {
  return labels[feature] || feature.replaceAll("_", " ");
}

function formatPercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  state.metadata = await response.json();
  renderMetrics();
  renderModel();
}

function renderMetrics() {
  const strip = document.querySelector("#metricStrip");
  strip.innerHTML = Object.values(state.metadata)
    .map(
      (model) => `
        <div class="metric">
          <strong>${formatPercent(model.metrics.accuracy)}</strong>
          <small>${model.title}</small>
        </div>
      `
    )
    .join("");
}

function renderModel() {
  const model = state.metadata[state.currentModel];
  document.querySelector("#modelTitle").textContent = model.title;
  document.querySelector("#modelNote").textContent = model.notes;
  document.querySelector("#predictionForm").innerHTML = model.features
    .map(
      (feature) => `
        <label>
          ${titleFor(feature)}
          <input
            type="number"
            name="${feature}"
            step="any"
            value="${model.defaults[feature]}"
            required
          />
        </label>
      `
    )
    .join("");
  document.querySelectorAll(".model-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.model === state.currentModel);
  });
  resetResult();
}

function resetResult() {
  const panel = document.querySelector("#resultPanel");
  panel.classList.remove("risk", "lower");
  document.querySelector("#resultLabel").textContent = "Enter values";
  document.querySelector("#resultProbability").textContent = "Probability will appear here.";
  document.querySelector("#gaugeFill").style.width = "0%";
}

function collectValues() {
  const formData = new FormData(document.querySelector("#predictionForm"));
  const values = {};
  for (const [key, value] of formData.entries()) {
    values[key] = Number(value);
  }
  return values;
}

async function predict(event) {
  event.preventDefault();
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: state.currentModel,
      values: collectValues(),
    }),
  });
  const result = await response.json();
  if (!response.ok) {
    alert(result.error || "Prediction failed");
    return;
  }

  const panel = document.querySelector("#resultPanel");
  const isRisk = result.probability >= 0.5;
  panel.classList.toggle("risk", isRisk);
  panel.classList.toggle("lower", !isRisk);
  document.querySelector("#resultLabel").textContent = result.label;
  document.querySelector("#resultProbability").textContent =
    `Model probability: ${formatPercent(result.probability)}. Confidence: ${formatPercent(result.confidence)}.`;
  document.querySelector("#gaugeFill").style.width = formatPercent(result.probability);
}

document.querySelectorAll(".model-tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.currentModel = button.dataset.model;
    renderModel();
  });
});

document.querySelector("#predictionForm").addEventListener("submit", predict);
document.querySelector("#resetButton").addEventListener("click", renderModel);
document.querySelector("#fillDefaults").addEventListener("click", renderModel);

loadMetadata();
