import { useEffect, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import axios from 'axios';

function App() {
  const [numberList, setNumberList] = useState(() => {
    const stored = localStorage.getItem('numberList');
    return stored ? JSON.parse(stored) : [];
  });

  const [prediction, setPrediction] = useState(null);
  const [correctCount, setCorrectCount] = useState(() => Number(localStorage.getItem('correctCount')) || 0);
  const [incorrectCount, setIncorrectCount] = useState(() => Number(localStorage.getItem('incorrectCount')) || 0);
  const [correctStreak, setCorrectStreak] = useState(0);
  const [incorrectStreak, setIncorrectStreak] = useState(0);
  const [maxCorrectStreak, setMaxCorrectStreak] = useState(() => Number(localStorage.getItem('maxCorrectStreak')) || 0);
  const [maxIncorrectStreak, setMaxIncorrectStreak] = useState(() => Number(localStorage.getItem('maxIncorrectStreak')) || 0);
  const [rollingAcc, setRollingAcc] = useState(0);

  const modelRef = useRef(null);
  const batchRef = useRef([]);
  const lastPredictionRef = useRef(null);
  const isTrainingRef = useRef(false);
  const trainCounter = useRef(0);
  const accHistory = useRef([]);
  const neuralAccTrack = useRef([]);
  const seenIssues = useRef(new Set());

  useEffect(() => {
    localStorage.setItem('numberList', JSON.stringify(numberList));
    localStorage.setItem('correctCount', correctCount);
    localStorage.setItem('incorrectCount', incorrectCount);
    localStorage.setItem('maxCorrectStreak', maxCorrectStreak);
    localStorage.setItem('maxIncorrectStreak', maxIncorrectStreak);
  }, [numberList, correctCount, incorrectCount, maxCorrectStreak, maxIncorrectStreak]);

  const extractFeatures = (n1, n2, n3) => {
    const norm = x => x / 9;
    const gap1 = Math.abs(n2 - n1) / 9;
    const gap2 = Math.abs(n3 - n2) / 9;
    const evenOdd = [n1 % 2, n2 % 2, n3 % 2];
    return [norm(n1), norm(n2), norm(n3), gap1, gap2, ...evenOdd];
  };

  const createModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [8], units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    return model;
  };

  useEffect(() => {
    tf.loadLayersModel('localstorage://hybrid-predictor')
      .then(model => {
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        modelRef.current = model;
      })
      .catch(() => {
        modelRef.current = createModel();
      });
  }, []);

  const saveModel = async () => {
    if (modelRef.current) {
      try {
        await modelRef.current.save('localstorage://hybrid-predictor');
      } catch (err) {
        console.error("Failed to save model:", err);
      }
    }
  };

  const trainBatch = async () => {
    if (isTrainingRef.current) return;
    const data = batchRef.current.slice(-200);
    if (data.length === 0) return;

    isTrainingRef.current = true;
    const xs = tf.tensor2d(data.map(d => d.input));
    const ys = tf.tensor2d(data.map(d => [d.label]));
    try {
      await modelRef.current.fit(xs, ys, { epochs: 20, batchSize: 16, shuffle: true, verbose: 0 });
      await saveModel();
    } catch (err) {
      console.error("Training error:", err);
    }
    xs.dispose();
    ys.dispose();
    isTrainingRef.current = false;
  };

  const markovPredict = (list) => {
    if (list.length < 3) return null;
    const key = `${list.at(-3).num}${list.at(-2).num}`;
    const transitions = {};
    for (let i = 2; i < list.length - 1; i++) {
      const k = `${list[i - 2].num}${list[i - 1].num}`;
      const next = list[i].num;
      if (!transitions[k]) transitions[k] = [];
      transitions[k].push(next);
    }
    const possible = transitions[key];
    if (!possible || possible.length === 0) return null;
    const avg = possible.reduce((a, b) => a + b, 0) / possible.length;
    return Math.round(avg);
  };

  const neuralPredict = async (n1, n2, n3) => {
    const input = tf.tensor2d([extractFeatures(n1, n2, n3)]);
    const result = modelRef.current.predict(input);
    const value = (await result.data())[0];
    input.dispose();
    result.dispose();
    return Math.round(Math.max(0, Math.min(9, value)));
  };

  const hybridPredict = async (n1, n2, n3, list) => {
    const markov = markovPredict(list);
    const neural = await neuralPredict(n1, n2, n3);

    neuralAccTrack.current.push(neural >= 5 ? 1 : 0);
    if (neuralAccTrack.current.length > 10) neuralAccTrack.current.shift();
    const neuralAccuracy = neuralAccTrack.current.filter(x => x === 1).length / neuralAccTrack.current.length;

    let final;
    const recentTrend = list.slice(-5).map(e => e.num >= 5 ? 1 : 0);
    const bigCount = recentTrend.reduce((a, b) => a + b, 0);

    if (markov === null || neuralAccuracy >= 0.6) {
      final = neural;
    } else if (markov === neural) {
      final = markov;
    } else {
      let weighted = (markov + neural) / 2;
      if (bigCount >= 4) weighted += 0.5;
      else if (bigCount <= 1) weighted -= 0.5;
      final = Math.round(Math.max(0, Math.min(9, weighted)));
    }

    return final;
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(
          "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
          { params: { ts: Date.now() } }
        );

        const latest = res.data.data.list[0];
        const num = Number(latest.number);
        const issue = latest.issueNumber;

        if (seenIssues.current.has(issue)) return;
        seenIssues.current.add(issue);

        const last = numberList[numberList.length - 1];
        const updatedList = [...numberList, {
          num,
          issue,
          predicted: lastPredictionRef.current ?? null
        }];
        setNumberList(updatedList);

        if (updatedList.length >= 4) {
          const n1 = updatedList.at(-4).num;
          const n2 = updatedList.at(-3).num;
          const n3 = updatedList.at(-2).num;
          const actual = num;

          const features = extractFeatures(n1, n2, n3);
          batchRef.current.push({ input: features, label: actual });
          trainCounter.current++;

          if (trainCounter.current >= 5) {
            trainCounter.current = 0;
            await trainBatch();
          }

          if (lastPredictionRef.current !== null) {
            const predictedBig = lastPredictionRef.current >= 5;
            const actualBig = actual >= 5;
            const isCorrect = predictedBig === actualBig;

            if (isCorrect) {
              setCorrectCount(c => c + 1);
              setCorrectStreak(s => {
                const newS = s + 1;
                setIncorrectStreak(0);
                setMaxCorrectStreak(m => Math.max(m, newS));
                return newS;
              });
            } else {
              setIncorrectCount(c => c + 1);
              setIncorrectStreak(s => {
                const newS = s + 1;
                setCorrectStreak(0);
                setMaxIncorrectStreak(m => Math.max(m, newS));
                return newS;
              });
            }

            accHistory.current.push(isCorrect ? 1 : 0);
            if (accHistory.current.length > 32) accHistory.current.shift();
            setRollingAcc(accHistory.current.reduce((a, b) => a + b, 0) / accHistory.current.length);
          }

          const predicted = await hybridPredict(n1, n2, n3, updatedList);
          setPrediction(predicted);
          lastPredictionRef.current = predicted;
        }
      } catch (e) {
        console.error("Prediction error:", e);
      }
    }, 4000);

    return () => clearInterval(interval);
  }, [numberList]);

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center p-6">
      <h1 className="text-2xl font-bold mb-4">Model 3 - Prediction Tracker</h1>

      <div className="mb-6 p-4 bg-gray-900 rounded shadow-md w-full max-w-md text-center text-xl">
        <span>Next Prediction: </span>
        {prediction !== null ? (
          <span className={prediction >= 5 ? 'text-green-400 font-bold text-2xl' : 'text-red-400 font-bold text-2xl'}>
            {prediction} - {prediction >= 5 ? 'Big' : 'Small'}
          </span>
        ) : (
          <span className="text-gray-500">Waiting...</span>
        )}
        <div className="mt-2 text-sm text-gray-400">
          ✅ Correct: <span className="text-green-400">{correctCount}</span> | ❌ Incorrect: <span className="text-red-400">{incorrectCount}</span>
        </div>
        <div className="text-sm text-gray-400 mt-1">
          Accuracy: {correctCount + incorrectCount > 0 ? (
            <span className="text-blue-400 font-bold">
              {Math.floor((correctCount / (correctCount + incorrectCount)) * 100)}%
            </span>
          ) : 'N/A'} <span> Total Moves: {numberList.length}</span>
        </div>
        <div className="text-sm text-gray-400 mt-1">
          Current Streak ➤ ✅ <span className="text-green-400">{correctStreak}</span> | ❌ <span className="text-red-400">{incorrectStreak}</span><br />
          Max Streak ➤ ✅ <span className="text-green-400">{maxCorrectStreak}</span> | ❌ <span className="text-red-400">{maxIncorrectStreak}</span>
        </div>
        <div className="text-sm text-yellow-400 mt-1">
          Rolling Accuracy (Last 32): <span className="font-bold">{(rollingAcc * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 w-full max-w-md overflow-y-auto max-h-[50vh]">
        {[...numberList].slice(-100).reverse().map((entry, index) => {
          const isCorrect = entry.predicted !== undefined && (entry.predicted >= 5) === (entry.num >= 5);
          const isBig = entry.num >= 5;

          return (
            <div
              key={index}
              className={`px-3 py-1.5 rounded text-sm flex gap-1.5 shadow-md bg-gray-800 ${isBig ? 'text-green-400 font-semibold' : 'text-red-500 font-semibold'}`}
            >
              <span className={`p-1 px-2 rounded ${isCorrect ? 'bg-gray-100 font-bold' : ''}`}>
                <span className={`${isCorrect ? 'text-black' : 'text-white'}`}>{entry.num} - </span>
                {isBig ? 'Big' : 'Small'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default App;
