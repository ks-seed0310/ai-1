const fs = require('fs');

// =========================================================================
// 1. 外部ファイル（語彙・学習データ・テストデータ）の読み込み
// =========================================================================
let VOCAB;
let trainData;
let testData;

const VOCAB_FILE = 'vocab.sys.json';
const STUDY_FILE = 'studyInput.sys.json';
const TEST_FILE  = 'translateTestData.sys.json';

// A. 語彙（VOCAB）
if (fs.existsSync(VOCAB_FILE)) {
  VOCAB = JSON.parse(fs.readFileSync(VOCAB_FILE, 'utf8'));
} else {
  VOCAB = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "私", "猫", "好き", "犬", "I", "cats", "like", ".", "dogs", "僕", "学習", "する", "明日", "食べ", "た", "米", "は", "が", "を", "に", "ついて"];
}

// B. 訓練データ（学習用）
if (fs.existsSync(STUDY_FILE)) {
  trainData = JSON.parse(fs.readFileSync(STUDY_FILE, 'utf8'));
} else {
  trainData = [
    { input: "私 好き 猫", target: ["I", "like", "cats", "."] },
    { input: "私 好き 犬", target: ["I", "like", "dogs", "."] }
  ];
}

// C. ⭕【新機能】外部テストデータ（抜き打ちテスト用）
if (fs.existsSync(TEST_FILE)) {
  console.log(`📂 外部テストファイル『${TEST_FILE}』を検出しました。読み込みます...`);
  testData = JSON.parse(fs.readFileSync(TEST_FILE, 'utf8'));
} else {
  console.log(`⚠️ 『${TEST_FILE}』が見つからないため、内部のデフォルトテストを使用します。`);
  testData = [
    { lang: "ja", text: "私 は 米 が 好き" },
    { lang: "ja", text: "僕 は 猫 に ついて 学習 する" }
  ];
}

// =========================================================================
// 2. スペック定義
// =========================================================================
const VOCAB_SIZE = 65536;  
const DIMENSIONS = 1024;  
const WEIGHTS_COUNT = VOCAB_SIZE * DIMENSIONS;
const LEARNING_RATE = 0.2; 

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
  console.log("🚀 真・全自動拡張＆双方向抜き打ちテスト型AIエンジンを起動します。");
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  console.log("⏳ 特訓を開始します。GitHub Actions のサーバーで処理中...");

  // 300回特訓
  for (let epoch = 1; epoch <= 300; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      for (let t = 0; t < data.target.length; t++) {
        const correctId = VOCAB.indexOf(data.target[t]);
        if (correctId === -1) continue;

        // 予測（Forward）
        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          inputIds.forEach((id, pos) => {
            const dimIdx = (id + pos + t) % DIMENSIONS;
            score += hWeightsMatrix[rowOffset + dimIdx];
          });
          resultScores[wordId] = score;
        }

        const hProbs = softmax(resultScores);

        // 誤差修正（Backward）
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

  console.log("\n✅ 特訓が完了しました！初めて見る文章で抜き打ちテスト（推論）を開始します...");

  // ⭕【超改良】日本語➔英語、英語➔日本語の両方に対応した翻訳テストアルゴリズム
  function testTranslation(lang, testInputText) {
    const testIds = tokenize(testInputText);
    const outputWords = [];

    // 文章の長さ（最大4語）だけ自動ループ
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
