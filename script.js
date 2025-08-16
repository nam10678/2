// script.js - fallback loader for tfjs_tie + tfjs_pb_N ensemble
const TIE_CANDIDATES = ["tfjs_tie/model.json","tfjs_model/model.json","model/model.json"];
const PB_CANDIDATES_BASE = ["tfjs_pb_1","tfjs_pb_2","tfjs_pb_3","tfjs_pb_4","tfjs_pb_5"];
const DEFAULT_TIE_THRESHOLD = 0.20; // nếu bạn có threshold khác, sửa ở đây

let tieModel = null;
let pbModels = [];
let tieThreshold = DEFAULT_TIE_THRESHOLD;
let history = [];

const ONE_HOT = { P: [1,0,0], B: [0,1,0], T: [0,0,1] };

async function tryLoad(path) {
  try {
    const m = await tf.loadLayersModel(path);
    console.log("Loaded model:", path);
    return m;
  } catch (e) {
    return null;
  }
}

async function loadModels() {
  // tie model - try candidates
  for (const p of TIE_CANDIDATES) {
    tieModel = await tryLoad(p);
    if (tieModel) break;
  }
  if (!tieModel) console.error("Tie model not found in:", TIE_CANDIDATES);

  // pb ensemble
  pbModels = [];
  for (const base of PB_CANDIDATES_BASE) {
    const path = `${base}/model.json`;
    const m = await tryLoad(path);
    if (m) pbModels.push(m);
  }
  if (pbModels.length === 0) console.warn("No PB models found.");
  console.log("Models loaded → tie:", !!tieModel, "pb_count:", pbModels.length);
}

// Helpers
function oneHotFromHistory(seq5) {
  const arr = seq5.flatMap(s => ONE_HOT[s] || [0,0,0]);
  return tf.tensor([arr], [1, 5, 3]);
}

async function predictFromHistory() {
  if (history.length < 5) return null;
  const seq5 = history.slice(-5);
  const inputTensor = oneHotFromHistory(seq5);

  // tie check
  if (tieModel) {
    try {
      const tiePred = await tieModel.predict(inputTensor).data();
      const tieProb = tiePred[0];
      if (tieProb >= tieThreshold) {
        inputTensor.dispose();
        return { label: "T", confidence: Number((tieProb*100).toFixed(2)) };
      }
    } catch (e) {
      console.error("tieModel predict error:", e);
    }
  }

  // PB ensemble
  if (pbModels.length > 0) {
    let sumP = 0;
    let got = 0;
    for (const m of pbModels) {
      try {
        const p = await m.predict(inputTensor).data();
        sumP += p[0];
        got++;
      } catch (e) {
        console.warn("pb model predict failed:", e);
      }
    }
    const avgP = got ? sumP / got : 0.5;
    const label = avgP >= 0.5 ? "P" : "B";
    inputTensor.dispose();
    return { label, confidence: Number(((avgP>=0.5?avgP:(1-avgP))*100).toFixed(2)) };
  }

  inputTensor.dispose();
  return { label: seq5[4], confidence: 0 };
}

// UI
function updateUIPrediction(pred) {
  const labelEl = document.getElementById("predicted-label");
  const confEl = document.getElementById("confidence");
  if (!labelEl || !confEl) return;
  if (!pred) {
    labelEl.textContent = "--";
    confEl.textContent = "--";
    labelEl.style.color = "";
    return;
  }
  labelEl.textContent = pred.label;
  confEl.textContent = pred.confidence + "%";
  labelEl.style.color = pred.label === "P" ? "green" : pred.label === "B" ? "red" : "gray";
}

async function addResult(ch) {
  if (!["P","B","T"].includes(ch)) return;
  history.push(ch);
  if (history.length > 200) history.shift();
  renderHistory();
  if (history.length >= 5) {
    const pred = await predictFromHistory();
    updateUIPrediction(pred);
  }
}

function resetHistory() {
  history = [];
  renderHistory();
  updateUIPrediction(null);
}

function renderHistory() {
  const el = document.getElementById("history");
  if (!el) return;
  el.innerHTML = history.map(h => `<span class="hist-item ${h}">${h}</span>`).join(" ");
}

// init
window.addEventListener("load", async () => {
  const stat = document.getElementById("model-status");
  if (stat) stat.textContent = "Loading models...";
  await loadModels();
  if (stat) stat.textContent = "Models loaded.";
  renderHistory();
});
