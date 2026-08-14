const fs = require('fs');

// =========================================================================
// 1. 外部ファイルの読み込み（1文字も溢さず自動追従）
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
const DIMENSIONS = 1024;  // 1024個のメーター（高エントロピー空間）
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

// ⭕【バグ修正版・高エントロピー展開】
// 単語IDを狂わせる「id + t」を完全に廃止。純粋に単語固有の美しい指紋ベクトルを作ります
function generateHighEntropyVector(inputIds) {
  const vector = new Float32Array(DIMENSIONS);
  inputIds.forEach((id, pos) => {
    for (let d = 0; d < DIMENSIONS; d++) {
      // 純粋に単語IDと文章内の位置（pos）だけで1024次元にサイン波を展開
      const angle = (id + pos) / Math.pow(10000, (2 * (d >> 1)) / DIMENSIONS);
      vector[d] += (d % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
    }
  });
  return vector;
}

async function run() {
  console.log("🚀 バグ修正完了。真・高エントロピーAIエンジンを起動します。");
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  console.log("⏳ 特訓を開始します。GitHub Actions のサーバーで処理中...");

  // 150回特訓
  for (let epoch = 1; epoch <= 150; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      // ⭕ 単語IDを狂わせない、ピュアな入力ベクトルを生成
      const hInputVector = generateHighEntropyVector(inputIds);

      for (let t = 0; t < data.target.length; t++) {
        const correctId = VOCAB.indexOf(data.target[t]);
        if (correctId === -1) continue;

        // 予測（Forward）：ステップ位置（t）の補正は、脳みその部屋のオフセット（dimIdx）側だけで行う！
        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          for (let d = 0; d < DIMENSIONS; d++) {
            // ステップ（t）に応じて読み出す部屋をスライドさせる正しい文法処理
            const dimIdx = (d + t) % DIMENSIONS;
            score += hInputVector[dimIdx] * hWeightsMatrix[rowOffset + d];
          }
          resultScores[wordId] = score;
        }

        const hProbs = softmax(resultScores);

        // 誤差修正（Backward）
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let error = hProbs[wordId];
          if (wordId === correctId) error -= 1.0;
          
          if (error !== 0) {
            const rowOffset = wordId * DIMENSIONS;
            for (let d = 0; d < DIMENSIONS; d++) {
              const dimIdx = (d + t) % DIMENSIONS;
              hWeightsMatrix[rowOffset + d] -= LEARNING_RATE * error * hInputVector[dimIdx];
            }
          }
        }
      }
    }
  }

  console.log("\n✅ 特訓が完了しました！抜き打ちテスト（推論）を開始します...");

  // テスト用翻訳関数（バグ修正版）
  function testTranslation(lang, testInputText) {
    const testIds = tokenize(testInputText);
    const outputWords = [];
    const hInputVector = generateHighEntropyVector(testIds);

    for (let t = 0; t < 4; t++) {
      const scores = new Float32Array(VOCAB_SIZE);
      for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
        let score = 0;
        const rowOffset = wordId * DIMENSIONS;
        for (let d = 0; d < DIMENSIONS; d++) {
          const dimIdx = (d + t) % DIMENSIONS;
          score += hInputVector[dimIdx] * hWeightsMatrix[rowOffset + d];
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

  // 外部JSONのテストデータに沿って全件テスト実行！
  for(const tData of testData) {
    await testTranslation(tData.lang, tData.text);
  }

  console.log("\n💾 賢くなった脳みそを『weights.bin』として爆速書き出し中...");
  const buffer = Buffer.from(hWeightsMatrix.buffer);
  fs.writeFileSync('weights.bin', buffer);
  console.log("📂 バイナリファイルの保存が完了しました！");
}

run();
