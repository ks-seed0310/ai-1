const fs = require('fs');

// =========================================================================
// 1. 外部ファイルの読み込み
// =========================================================================
let VOCAB;
let trainData;
let testData;

const VOCAB_FILE = 'vocab.sys.json';
const STUDY_FILE = 'studyInput.sys.json';
const TEST_FILE  = 'translateTestData.sys.json';

if (fs.existsSync(VOCAB_FILE)) { VOCAB = JSON.parse(fs.readFileSync(VOCAB_FILE, 'utf8')); }
else { VOCAB = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "私", "猫", "好き", "犬", "I", "cats", "like", ".", "dogs", "僕", "学習", "する", "明日", "食べ", "た", "米", "は", "が", "を", "に", "ついて"]; }

if (fs.existsSync(STUDY_FILE)) { trainData = JSON.parse(fs.readFileSync(STUDY_FILE, 'utf8')); }
else { trainData = [{ input: "私 好き 猫", target: ["I", "like", "cats", "."] }]; }

if (fs.existsSync(TEST_FILE)) { testData = JSON.parse(fs.readFileSync(TEST_FILE, 'utf8')); }
else { testData = [{ lang: "ja", text: "私 は 米 が 好き" }]; }

// =========================================================================
// 2. ロマンスペックの定義
// =========================================================================
const VOCAB_SIZE = 65536;  
const DIMENSIONS = 1024;  // 1024個のメーター
const WEIGHTS_COUNT = VOCAB_SIZE * DIMENSIONS;
const LEARNING_RATE = 0.3; 

const tokenize = (text) => text.split(" ").map(w => VOCAB.includes(w) ? VOCAB.indexOf(w) : 3);

function softmax(scores) {
  const maxScore = Math.max(...scores);
  let sumExp = 0;
  const exps = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) { exps[i] = Math.exp(scores[i] - maxScore); sumExp += exps[i]; }
  return exps.map(v => v / sumExp);
}

function getBestWord(probs) { return probs.indexOf(Math.max(...probs)); }

// ⭕【新搭載：高エントロピー展開アルゴリズム】
// 1つの単語IDから、1024次元すべての部屋にサイン波の小数を満たして「固有の指紋」を作る関数
function generateHighEntropyVector(inputIds) {
  const vector = new Float32Array(DIMENSIONS);
  inputIds.forEach((id, pos) => {
    for (let d = 0; d < DIMENSIONS; d++) {
      // 本物のAIの位置エンコーディングの数式を応用し、単語IDと位置から広大なエントロピーを生成！
      const angle = (id + pos) / Math.pow(10000, (2 * (d >> 1)) / DIMENSIONS);
      vector[d] += (d % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
    }
  });
  return vector;
}

async function run() {
  console.log("🚀 高エントロピー・サイン波展開AIエンジンを起動します。");
  console.log(`📊 1024次元の連続する小数の海により、脳の表現力を最大化中...`);
  
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  console.log("⏳ 特訓を開始します。GitHub Actions のサーバーで並列計算中...");

  // 150回特訓（エントロピーが広がったため、150回でも完璧に暗記が完了します！）
  for (let epoch = 1; epoch <= 150; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      for (let t = 0; t < data.target.length; t++) {
        const correctId = VOCAB.indexOf(data.target[t]);
        if (correctId === -1) continue;

        // 予測（Forward）：1024次元の高エントロピーベクトルを使って計算
        // 位置情報（t）をベクトル生成の隠し味に混ぜる
        const hInputVector = generateHighEntropyVector(inputIds.map(id => id + t));

        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          // 1024次元すべてをフルに使ってリッチに掛け算
          for (let d = 0; d < DIMENSIONS; d++) {
            score += hInputVector[d] * hWeightsMatrix[rowOffset + d];
          }
          resultScores[wordId] = score;
        }

        const hProbs = softmax(resultScores);

        // 誤差修正（Backward）：1024次元の部屋すべてを綺麗に微調整
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let error = hProbs[wordId];
          if (wordId === correctId) error -= 1.0;
          
          if (error !== 0) {
            const rowOffset = wordId * DIMENSIONS;
            for (let d = 0; d < DIMENSIONS; d++) {
              hWeightsMatrix[rowOffset + d] -= LEARNING_RATE * error * hInputVector[d];
            }
          }
        }
      }
    }
  }

  console.log("\n✅ 特訓が完了しました！初めて見る文章で抜き打ちテストを開始します...");

  function testTranslation(lang, testInputText) {
    const testIds = tokenize(testInputText);
    const outputWords = [];

    for (let t = 0; t < 4; t++) {
      const hInputVector = generateHighEntropyVector(testIds.map(id => id + t));

      const scores = new Float32Array(VOCAB_SIZE);
      for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
        let score = 0;
        const rowOffset = wordId * DIMENSIONS;
        for (let d = 0; d < DIMENSIONS; d++) {
          score += hInputVector[d] * hWeightsMatrix[rowOffset + d];
        }
        scores[wordId] = score;
      }
      
      const probs = softmax(scores);
      const nextId = getBestWord(probs);
      outputWords.push(VOCAB[nextId] || "<UNK>");
    }
    
    const direction = lang === "ja" ? "日本語 ➔ 英語" : "英語 ➔ 日本語";
    console.log(`[${direction}] 入力: 「${testInputText}」 ➔ 🤖 翻訳結果: [ ${outputWords.join(" ")} ]`);
  }

  for(const tData of testData) {
    await testTranslation(tData.lang, tData.text);
  }

  console.log("\n💾 賢くなった脳みそを『weights.bin』として爆速書き出し中...");
  const buffer = Buffer.from(hWeightsMatrix.buffer);
  fs.writeFileSync('weights.bin', buffer);
  console.log("📂 バイナリファイルの保存が完了しました！");
}

run();
