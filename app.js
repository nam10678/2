
let history = [];
let model;
let chart;

const labelMap = { P: 0, B: 1, T: 2 };
const reverseLabelMap = ['P', 'B', 'T'];

async function loadModel() {
  try {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("✅ Model loaded");
  } catch (error) {
    console.error("❌ Failed to load model:", error);
  }
}
loadModel();

function addResult(result) {
  history.push(result);
  if (history.length > 100) history.shift();
  updateChart();
  if (model && history.length >= 5) predict();
  else if (!model) {
    console.warn("⚠️ Model chưa sẵn sàng");
  }
}

function encodeInput(sequence) {
  return sequence.map(r => {
    const arr = [0, 0, 0];
    arr[labelMap[r]] = 1;
    return arr;
  }).flat();
}

async function predict() {
  const input = encodeInput(history.slice(-5));
  const inputTensor = tf.tensor([input]);
  const prediction = model.predict(inputTensor);
  const data = await prediction.data();

  const maxIdx = data.indexOf(Math.max(...data));
  const confidence = (data[maxIdx] * 100).toFixed(2);

  document.getElementById('predicted-label').textContent = reverseLabelMap[maxIdx];
  document.getElementById('confidence').textContent = confidence;

  const color = maxIdx === 0 ? 'blue' : maxIdx === 1 ? 'red' : 'gray';
  document.getElementById('predicted-label').style.color = color;
}

function resetHistory() {
  history = [];
  document.getElementById('predicted-label').textContent = "--";
  document.getElementById('confidence').textContent = "--";
  updateChart();
}

function updateChart() {
  const ctx = document.getElementById('historyChart').getContext('2d');
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: history.map((_, i) => i + 1),
      datasets: [{
        label: 'Lịch sử',
        data: history.map(r => labelMap[r]),
        backgroundColor: history.map(r => r === 'P' ? 'blue' : r === 'B' ? 'red' : 'gray'),
        borderColor: '#000',
        borderWidth: 1,
        pointRadius: 6
      }]
    },
    options: {
      scales: {
        y: {
          ticks: {
            callback: function (val) {
              return reverseLabelMap[val];
            },
            stepSize: 1,
            min: 0,
            max: 2
          }
        }
      }
    }
  });
}
