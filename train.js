const fs = require('fs');

// =========================================================================
// 1. 【バグ防止】外部ファイル読み込み ➔ なければ内部プリセットを使うアルゴリズム
// =========================================================================
let VOCAB;
let trainData;

const VOCAB_FILE = 'vocab.sys.json';
const STUDY_FILE = 'studyInput.sys.json';

// A. 語彙（VOCAB）のチェックと読み込み
if (fs.existsSync(VOCAB_FILE)) {
  console.log(`📂 外部ファイル『${VOCAB_FILE}』を検出しました。読み込みます...`);
  VOCAB = JSON.parse(fs.readFileSync(VOCAB_FILE, 'utf8'));
} else {
  console.log(`⚠️ 『${VOCAB_FILE}』が見つからないため、内部のプリセットデータを使用します。`);
  VOCAB = [
    "<PAD>", "<BOS>", "<EOS>", "<UNK>", 
    "私", "猫", "好き", "犬",          
    "I", "cats", "like", ".", "dogs"    
  ];
}

// B. 訓練データ（trainData）のチェックと読み込み
if (fs.existsSync(STUDY_FILE)) {
  console.log(`📂 外部ファイル『${STUDY_FILE}』を検出しました。読み込みます...`);
  trainData = JSON.parse(fs.readFileSync(STUDY_FILE, 'utf8'));
} else {
  console.log(`⚠️ 『${STUDY_FILE}』が見つからないため、内部のプリセットデータを使用します。`);
  trainData = [
    { input: "私 好き 猫", target: ["I", "like", "cats", "."] },
    { input: "私 好き 犬", target: ["I", "like", "dogs", "."] } 
  ];
}

// =========================================================================
// 2. ロマンスペックの自動計算（語彙数が増えてもここが自動追従します）
// =========================================================================
const VOCAB_SIZE = 65536;  // 6万5千語の最大部屋数はキープ
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
  console.log("🚀 真・外部ファイル対応型AIエンジン（正順・高速化版）を起動します。");
  console.log(`📊 現在の処理スペック: 登録語彙数 [ ${VOCAB.length} ] / 登録学習データ [ ${trainData.length} 件 ]`);
  
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  console.log("⏳ 特訓を開始します。GitHub Actions のサーバーで処理中...");

  // 150回特訓
  for (let epoch = 1; epoch <= 150; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      for (let t = 0; t < data.target.length; t++) {
        const correctId = VOCAB.indexOf(data.target[t]);

        if (correctId === -1) {
          // もしターゲットに登録外の単語があった場合のセーフティ
          continue; 
        }

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

  console.log("\n✅ 特訓が完了しました！学習成果をテストします...");

  // テスト（登録されている全データで翻訳確認する賢い仕様に変更）
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
      outputWords.push(VOCAB[nextId] || "<UNK>");
    }
    console.log(`入力: 「${testInputText}」 ➔ 🤖 翻訳結果: [ ${outputWords.join(" ")} ]`);
  }

  // 読み込まれた trainData の中身を自動で全件テストする
  for(const data of trainData) {
    await testTranslation(data.input);
  }

  console.log("\n💾 賢くなった脳みそを『weights.bin』として爆速書き出し中...");
  const buffer = Buffer.from(hWeightsMatrix.buffer);
  fs.writeFileSync('weights.bin', buffer);
  console.log("📂 バイナリファイルの保存が完了しました！");
}

run();
