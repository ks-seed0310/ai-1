const fs = require('fs');

// =========================================================================
// 1. 共通辞書とロマンスペックの設定
// =========================================================================
const VOCAB = [
  "<PAD>", "<BOS>", "<EOS>", "<UNK>", 
  "私", "猫", "好き", "犬",          
  "I", "cats", "like", ".", "dogs"    
];
const VOCAB_SIZE = 65536;  
const DIMENSIONS = 1024;  
const WEIGHTS_COUNT = VOCAB_SIZE * DIMENSIONS;
const LEARNING_RATE = 0.1;

const trainData = [
  { input: "私 好き 猫", target: ["I", "like", "cats", "."] },
  { input: "私 好き 犬", target: ["I", "like", "dogs", "."] } 
];

const tokenize = (text) => text.split(" ").map(w => VOCAB.includes(w) ? VOCAB.indexOf(w) : 3);

function softmax(scores) {
  const maxScore = Math.max(...scores);
  let sumExp = 0;
  const exps = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) { exps[i] = Math.exp(scores[i] - maxScore); sumExp += exps[i]; }
  return exps.map(v => v / sumExp);
}

function getBestWord(probs) {
  return probs.indexOf(Math.max(...probs));
}

async function run() {
  console.log("🚀 GitHub Actions 上で自動学習エンジンが起動しました。");
  console.log("🧠 6,700万個の脳みそデータ（約268MB）をメモリに展開中...");

  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  console.log("⏳ 特訓を開始します。GitHubのサーバーが代わりにフル稼働しています...");

  for (let epoch = 1; epoch <= 200; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      let currentHistoryIds = [];

      // 1語ずつ、それまでの「英語の出力履歴」を積み上げながら連動して学習
      for (let t = 0; t < data.target.length; t++) {
        const hInputVector = new Float32Array(DIMENSIONS);
        inputIds.forEach(id => { if(id < DIMENSIONS) hInputVector[id] = 1.0; });
        
        // 英語の履歴ベクトルもONにする
        currentHistoryIds.forEach(id => { if(id < DIMENSIONS) hInputVector[id] = 1.0; });

        const correctWord = data.target[t];
        const correctId = VOCAB.indexOf(correctWord);

        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          for (let i = 0; i < DIMENSIONS; i++) { score += hInputVector[i] * hWeightsMatrix[rowOffset + i]; }
          resultScores[wordId] = score;
        }

        const hProbs = softmax(resultScores);

        const wordErrors = new Float32Array(VOCAB_SIZE);
        for (let i = 0; i < VOCAB_SIZE; i++) {
          wordErrors[i] = hProbs[i];
          if (i === correctId) wordErrors[i] -= 1.0;
        }

        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          const error = wordErrors[wordId];
          const rowOffset = wordId * DIMENSIONS;
          if (error !== 0) {
            for (let i = 0; i < DIMENSIONS; i++) { hWeightsMatrix[rowOffset + i] -= LEARNING_RATE * error * hInputVector[i]; }
          }
        }

        currentHistoryIds.push(correctId);
      }
    }
  }

  console.log("\n✅ GitHub Actions での自動学習が200回無事に完了しました！");
  console.log("🧪 学習成果をテストします...");

  // ⭕【バグ修正】テスト時も自分で出力した単語の履歴を1語ずつ脳にフィードバックする
  function testTranslation(testInputText) {
    const testInputIds = tokenize(testInputText);
    let testHistoryIds = []; 
    const outputWords = [];

    for (let step = 0; step < 4; step++) {
      const hInputVector = new Float32Array(DIMENSIONS);
      testInputIds.forEach(id => { if(id < DIMENSIONS) hInputVector[id] = 1.0; });
      
      // AIが「今までに自分で出力した言葉」を履歴としてメーターにON！
      testHistoryIds.forEach(id => { if(id < DIMENSIONS) hInputVector[id] = 1.0; });

      const scores = new Float32Array(VOCAB_SIZE);
      for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
        let score = 0;
        const rowOffset = wordId * DIMENSIONS;
        for (let i = 0; i < DIMENSIONS; i++) { score += hInputVector[i] * hWeightsMatrix[rowOffset + i]; }
        scores[wordId] = score;
      }
      
      const probs = softmax(scores);
      const nextId = getBestWord(probs);
      
      outputWords.push(VOCAB[nextId]);
      testHistoryIds.push(nextId); // 自分の出した言葉を記憶する
    }
    console.log(`入力: 「${testInputText}」 ➔ 🤖 翻訳結果: [ ${outputWords.join(" ")} ]`);
  }

  await testTranslation("私 好き 猫");
  await testTranslation("私 好き 犬");

  console.log("\n💾 賢くなった脳みそ（重みデータ）を『weights.json』として書き出し中...");
  const weightsArray = Array.from(hWeightsMatrix);
  fs.writeFileSync('weights.json', JSON.stringify(weightsArray));
  console.log("📂 ファイルの書き出しが成功しました。");
}

run();
