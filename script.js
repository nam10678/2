// ================= Baccarat Predictor Script =================

// Load candidates models
const TIE_CANDIDATES = ["./tfjs_tie/model.json"];
const PB_CANDIDATES_BASE = [
  "./tfjs_pb_1",
  "./tfjs_pb_2",
  "./tfjs_pb_3",
  "./tfjs_pb_4",
  "./tfjs_pb_5"
];

const DEFAULT_TIE_THRESHOLD = 0.20;
let tieModel = null;
let pbModels = [];
let tieThreshold = DEFAULT_TIE_THRESHOLD;
let history = [];

// --------- Load models ---------
async function loadModels() {
  console.log("Loading models...");

  // Try Tie model
  for (let url of TIE_CANDIDATES) {
    try {
      tieModel = await tf.loadLayersModel(url);
      console.log("✅ Tie model loaded:", url);
      break;
    } catch (err) {
      console.warn("Tie model load failed:", url, err.message);
    }
  }

  // Try PB models
  for (let base of PB_CANDIDATES_BASE) {
    try {
      let m = await tf.loadLayersModel(`${base}/model.json`);
      pbModels.push(m);
      console.log("✅ PB model loaded:", base);
    } catch (err) {
      console.warn("PB model load failed:", base, err.message);
    }
  }

  console.log(`Models loaded -> Tie: ${tieModel ? "ok" : "none"} | PB count: ${pbModels.length}`);
}

// --------- Encode input ---------
function encodeInput(seq) {
  const map = { "P": [1, 0, 0], "B": [0, 1, 0], "T": [0, 0, 1] };
  return tf.tensor([seq.map(c => map[c])], [1, 5, 3]);
}

// --------- Predict ---------
async function predictNext(seq) {
  if (seq.length < 5) return { label: "?", probs: [0, 0, 0] };

  let x = encodeInput(seq);

  // Tie prediction
  let tieProb = 0.0;
  if (tieModel) {
    tieProb = (await tieModel.predict(x).data())[0];
  }

  // P/B prediction
  let pbProb = 0.5;
  if (pbModels.length > 0) {
    let probs = [];
    for (let m of pbModels) {
      let p = (await m.predict(x).data())[0];
      probs.push(p);
    }
    pbProb = probs.reduce((a, b) => a + b, 0) / probs.length;
  }

  // Combine
  let probs = [0, 0, 0]; // [P,B,T]
  if (tieProb >= tieThreshold) {
    probs = [0, 0, tieProb];
  } else {
    probs = [pbProb * (1 - tieProb), (1 - pbProb) * (1 - tieProb), tieProb];
  }

  let label = ["P", "B", "T"][probs.indexOf(Math.max(...probs))];
  return { label, probs };
}

// --------- UI handlers ---------
function addResult(res) {
  history.push(res);
  if (history.length > 5) history.shift();
  updateHistoryUI();
}

async function updatePrediction() {
  if (history.length < 5) {
    document.getElementById("prediction").innerText = "Need 5 results first.";
    return;
  }

  let { label, probs } = await predictNext(history.slice(-5));
  let txt = `Predict: ${label} | P=${(probs[0] * 100).toFixed(1)}%  B=${(probs[1] * 100).toFixed(1)}%  T=${(probs[2] * 100).toFixed(1)}%`;
  document.getElementById("prediction").innerText = txt;
}

function updateHistoryUI() {
  document.getElementById("history").innerText = "History: " + history.join(", ");
}

// --------- Button bindings ---------
function setupButtons() {
  document.getElementById("btnP").onclick = () => { addResult("P"); updatePrediction(); };
  document.getElementById("btnB").onclick = () => { addResult("B"); updatePrediction(); };
  document.getElementById("btnT").onclick = () => { addResult("T"); updatePrediction(); };
  document.getElementById("btnReset").onclick = () => { history = []; updateHistoryUI(); document.getElementById("prediction").innerText = ""; };
}

// --------- Init ---------
window.onload = async () => {
  await loadModels();
  setupButtons();
  updateHistoryUI();
  document.getElementById("prediction").innerText = "Ready.";
};
