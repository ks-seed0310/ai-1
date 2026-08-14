const fs = require('fs');

const VOCAB = [
  "<PAD>", "<BOS>", "<EOS>", "<UNK>", 
  "私", "猫", "好き", "犬",          
  "I", "cats", "like", ".", "dogs"    
];
const VOCAB_SIZE = 65536;  
const DIMENSIONS = 1024;  
const WEIGHTS_COUNT = VOCAB_SIZE * DIMENSIONS;
const LEARNING_RATE = 0.2; 

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

function getBestWord(probs) { return probs.indexOf(Math.max(...probs)); }

async function run() {
  console.log("🚀 真・全自動拡張型AIエンジン（正順・高速化版）を起動します。");
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  // 150回特訓
  for (let epoch = 1; epoch <= 150; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      // ⭕ 順番（t）をインデックス計算に正しく連動させる
      for (let t = 0; t < data.target.length; t++) {
        const correctId = VOCAB.indexOf(data.target[t]);

        // 予測（Forward）：入力された単語の位置（t）をオフセットに組み込む
        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          inputIds.forEach((id, pos) => {
            // 単語固有のIDと、その位置（posとt）を掛け合わせて部屋を完全に分離
            const dimIdx = (id + pos + t) % DIMENSIONS;
            score += hWeightsMatrix[rowOffset + dimIdx];
          });
          resultScores[wordId] = score;
        }

        const hProbs = softmax(resultScores);

        // 誤差修正（Backward）：位置情報を連動させて正確に脳みそを書き換える
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let error = hProbs[wordId];
          if (wordId === correctId) error -= 1.0;
          
          if (error !== 0) {
            const rowOffset = wordId * DIMENSIONS;
            inputIds.forEach((id, pos) => {
              const dimIdx = (id + pos + t) % DIMENSIONS;
              hWeightsMatrix[rowOffset + dimIdx] -= LEARNING_RATE * error;
            });
          }
        }
      }
    }
  }

  console.log("\n✅ 特訓が完了しました！学習成果をテストします...");

  function testTranslation(testInputText) {
    const testIds = tokenize(testInputText);
    const outputWords = [];

    for (let t = 0; t < 4; t++) {
      const scores = new Float32Array(VOCAB_SIZE);
      for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
        let score = 0;
        const rowOffset = wordId * DIMENSIONS;
        testIds.forEach((id, pos) => {
          const dimIdx = (id + pos + t) % DIMENSIONS;
          score += hWeightsMatrix[rowOffset + dimIdx];
        });
        scores[wordId] = score;
      }
      
      const probs = softmax(scores);
      const nextId = getBestWord(probs);
      outputWords.push(VOCAB[nextId]);
    }
    console.log(`入力: 「${testInputText}」 ➔ 🤖 翻訳結果: [ ${outputWords.join(" ")} ]`);
  }

  await testTranslation("私 好き 猫");
  await testTranslation("私 好き 犬");

  console.log("\n💾 賢くなった脳みそを『weights.bin』として爆速書き出し中...");
  const buffer = Buffer.from(hWeightsMatrix.buffer);
  fs.writeFileSync('weights.bin', buffer);
  console.log("📂 バイナリファイルの保存が完了しました！");
}

run();
