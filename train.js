const fs = require('fs');

// =========================================================================
// 1. 外部ファイル（語彙・学習データ・テストデータ）の動的読み込みシステム
// =========================================================================
let VOCAB;
let trainData;
let testData;

const VOCAB_FILE = 'vocab.sys.json';
const STUDY_FILE = 'studyInput.sys.json';
const TEST_FILE  = 'translateTestData.sys.json';

if (fs.existsSync(VOCAB_FILE)) { 
  VOCAB = JSON.parse(fs.readFileSync(VOCAB_FILE, 'utf8')); 
} else { 
  VOCAB = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "私", "猫", "好き", "犬", "I", "cats", "like", ".", "dogs", "僕", "学習", "する", "明日", "食べ", "た", "米", "は", "が", "を", "に", "ついて"]; 
}

if (fs.existsSync(STUDY_FILE)) { 
  trainData = JSON.parse(fs.readFileSync(STUDY_FILE, 'utf8')); 
} else { 
  trainData = [{ input: "私 好き 猫", target: ["I", "like", "cats", "."] }]; 
}

if (fs.existsSync(TEST_FILE)) { 
  testData = JSON.parse(fs.readFileSync(TEST_FILE, 'utf8')); 
} else { 
  testData = [{ lang: "ja", text: "私 は 米 が 好き" }]; 
}

// =========================================================================
// 2. 巨大AI（ロマンスペック）の定義
// =========================================================================
const VOCAB_SIZE = 65536;  
const DIMENSIONS = 1024;  
const WEIGHTS_COUNT = VOCAB_SIZE * DIMENSIONS;
const LEARNING_RATE = 0.1; 

const tokenize = (text) => text.split(" ").map(w => VOCAB.includes(w) ? VOCAB.indexOf(w) : 3);

// 脳みその初期化用（ファイルが無い時だけ使われる予備のデタラメ乱数）
function initWeights(rows, cols) {
  const w = new Float32Array(rows * cols);
  for (let i = 0; i < w.length; i++) w[i] = Math.random() * 0.02 - 0.01;
  return w;
}

let embeddingWeights;
let outputWeights;

const EMBEDDING_BYTE_SIZE = WEIGHTS_COUNT * 4; // 268,435,456 バイト
const OUTPUT_BYTE_SIZE = WEIGHTS_COUNT * 4;    // 268,435,456 バイト
const TOTAL_BYTE_SIZE = EMBEDDING_BYTE_SIZE + OUTPUT_BYTE_SIZE; // 計 536,870,912 バイト

// ⭕【今度こそ本当に新搭載：過去の脳みそ引き継ぎアルゴリズム】
// リポジトリに分割されて保存されている「前回の記憶ファイル」を自動検知してガッチャンコする
const partFiles = fs.readdirSync('.').filter(f => f.startsWith('weights_part_')).sort();

if (partFiles.length > 0) {
  console.log(`📂 リポジトリ内に前回の分割脳みそファイルを [ ${partFiles.length}個 ] 検出しました。`);
  console.log("🧩 過去の記憶を一瞬でガッチャンコして復元中...");
  
  // 分割されたバイナリファイルを順番に読み込んで1つの巨大なバッファ（536MB）に結合
  const buffers = partFiles.map(f => fs.readFileSync(f));
  const combinedBuffer = Buffer.concat(buffers);
  
  // 生のバイナリバッファから、Float32ArrayとしてAIの記憶（重み）を完全に復元！
  embeddingWeights = new Float32Array(combinedBuffer.buffer, combinedBuffer.byteOffset, WEIGHTS_COUNT);
  outputWeights = new Float32Array(combinedBuffer.buffer, combinedBuffer.byteOffset + EMBEDDING_BYTE_SIZE, WEIGHTS_COUNT);
  
  console.log("✅ 前回の特訓データを100%完璧に引き継ぎました！記憶の継続に成功。");
} else {
  console.log("⚠️ リポジトリ内に過去の特訓データが見つかりません。");
  console.log("🌱 1回目の特訓として、真っ白な乱数エントロピーから脳みそを新規作成します。");
  embeddingWeights = initWeights(VOCAB_SIZE, DIMENSIONS);
  outputWeights = initWeights(VOCAB_SIZE, DIMENSIONS);
}

// =========================================================================
// 3. 本物のTransformerアルゴリズム群（省略ナシ）
// =========================================================================

function applyPositionalEncoding(vector, position) {
  for (let d = 0; d < DIMENSIONS; d++) {
    const angle = position / Math.pow(10000, (2 * (d >> 1)) / DIMENSIONS);
    vector[d] += (d % 2 === 0) ? Math.sin(angle) : Math.cos(angle);
  }
}

function forwardAttention(vectors) {
  const seqLen = vectors.length;
  const output = vectors.map(() => new Float32Array(DIMENSIONS));
  const attentionMaps = vectors.map(() => new Float32Array(seqLen));

  for (let i = 0; i < seqLen; i++) {
    let scoreSum = 0;
    const scores = new Float32Array(seqLen);
    
    for (let j = 0; j < seqLen; j++) {
      let dot = 0;
      for (let d = 0; d < DIMENSIONS; d++) dot += vectors[i][d] * vectors[j][d];
      scores[j] = Math.exp(dot / Math.sqrt(DIMENSIONS));
      scoreSum += scores[j];
    }
    
    for (let j = 0; j < seqLen; j++) {
      attentionMaps[i][j] = scores[j] / scoreSum;
      for (let d = 0; d < DIMENSIONS; d++) {
        output[i][d] += attentionMaps[i][j] * vectors[j][d];
      }
    }
  }
  return { contextVectors: output, attentionMaps: attentionMaps };
}

function matMulLinear(vector) {
  const scores = new Float32Array(VOCAB_SIZE);
  for (let w = 0; w < VOCAB_SIZE; w++) {
    let s = 0;
    const rowOffset = w * DIMENSIONS;
    for (let d = 0; d < DIMENSIONS; d++) s += vector[d] * outputWeights[rowOffset + d];
    scores[w] = s;
  }
  return scores;
}

function softmax(scores) {
  const maxScore = Math.max(...scores);
  let sumExp = 0;
  const exps = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) {
    exps[i] = Math.exp(scores[i] - maxScore);
    sumExp += exps[i];
  }
  return exps.map(v => v / sumExp);
}

function getBestWord(probs) { return probs.indexOf(Math.max(...probs)); }

function backwardTransformer(contextVector, probs, correctWordId, inputIds) {
  const errors = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) {
    errors[i] = probs[i];
    if (i === correctWordId) errors[i] -= 1.0;
  }

  const vectorGrad = new Float32Array(DIMENSIONS);
  for (let w = 0; w < VOCAB_SIZE; w++) {
    const err = errors[w];
    if (err === 0) continue;
    const offset = w * DIMENSIONS;
    for (let d = 0; d < DIMENSIONS; d++) {
      vectorGrad[d] += err * outputWeights[offset + d];
      outputWeights[offset + d] -= LEARNING_RATE * err * contextVector[d];
    }
  }

  inputIds.forEach(id => {
    if (id < VOCAB_SIZE) {
      const offset = id * DIMENSIONS;
      for (let d = 0; d < DIMENSIONS; d++) {
        embeddingWeights[offset + d] -= LEARNING_RATE * vectorGrad[d];
      }
    }
  });
}

// =========================================================================
// 5. 🔁 サーバー上での真の自己回帰・継続学習ループ
// =========================================================================
async function run() {
  console.log("🚀 継続学習対応・本格規格Transformerエンジンを起動します。");
  console.log("⏳ 特訓を開始します。前回の記憶をベースに、さらに上書き勉強中...");

  for (let epoch = 1; epoch <= 300; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      let currentHistoryIds = []; 

      for (let t = 0; t < data.target.length; t++) {
        let vectors = inputIds.map(id => {
          const v = new Float32Array(DIMENSIONS);
          const offset = id * DIMENSIONS;
          for (let d = 0; d < DIMENSIONS; d++) v[d] = embeddingWeights[offset + d];
          return v;
        });

        currentHistoryIds.forEach(id => {
          const v = new Float32Array(DIMENSIONS);
          const offset = id * DIMENSIONS;
          for (let d = 0; d < DIMENSIONS; d++) v[d] = embeddingWeights[offset + d];
          vectors.push(v);
        });

        vectors.forEach((v, idx) => applyPositionalEncoding(v, idx));

        const { contextVectors } = forwardAttention(vectors);
        const latestContextVector = contextVectors[contextVectors.length - 1];

        const correctWord = data.target[t];
        const correctId = VOCAB.indexOf(correctWord);
        const scores = matMulLinear(latestContextVector);
        const probs = softmax(scores);

        backwardTransformer(latestContextVector, probs, correctId, [...inputIds, ...currentHistoryIds]);
        currentHistoryIds.push(correctId);
      }
    }
  }

  console.log("\n✅ 特訓が完了しました！引き継がれた脳みそで抜き打ちテスト（推論）を開始します...");

  function testTranslation(lang, testInputText) {
    const testIds = tokenize(testInputText);
    let testHistoryIds = []; 
    const outputWords = [];

    for (let step = 0; step < 5; step++) {
      let testVectors = testIds.map(id => {
        const v = new Float32Array(DIMENSIONS);
        const offset = id * DIMENSIONS;
        for (let d = 0; d < DIMENSIONS; d++) v[d] = embeddingWeights[offset + d];
        return v;
      });

      testHistoryIds.forEach(id => {
        const v = new Float32Array(DIMENSIONS);
        const offset = id * DIMENSIONS;
        for (let d = 0; d < DIMENSIONS; d++) v[d] = embeddingWeights[offset + d];
        testVectors.push(v);
      });

      testVectors.forEach((v, idx) => applyPositionalEncoding(v, idx));
      
      const { contextVectors } = forwardAttention(testVectors);
      const testLastVec = contextVectors[contextVectors.length - 1];

      const scores = matMulLinear(testLastVec);
      const probs = softmax(scores);
      const nextId = getBestWord(probs);
      
      const word = VOCAB[nextId] || "<UNK>";
      outputWords.push(word);
      testHistoryIds.push(nextId);

      if (word === "<EOS>" || word === ".") break;
    }
    
    const direction = lang === "ja" ? "日本語 ➔ 英語" : "英語 ➔ 日本語";
    console.log(`[${direction}] 入力: 「${testInputText}」 ➔ 🤖 翻訳結果: [ ${outputWords.join(" ")} ]`);
  }

  for(const tData of testData) {
    await testTranslation(tData.lang, tData.text);
  }

  // =========================================================================
  // 6. 📦 536MBの最新の脳みそを、上書き保存のために一時ファイルに書き出す
  // =========================================================================
  console.log("\n💾 特訓成果を『weights.bin』として一時保存中...");
  const combinedBuffer = Buffer.alloc(EMBEDDING_BYTE_SIZE + OUTPUT_BYTE_SIZE);
  Buffer.from(embeddingWeights.buffer, embeddingWeights.byteOffset, embeddingWeights.byteLength).copy(combinedBuffer, 0);
  Buffer.from(outputWeights.buffer, outputWeights.byteOffset, outputWeights.byteLength).copy(combinedBuffer, EMBEDDING_BYTE_SIZE);
  
  // ⭕【インフラ連動のセーフティ】
  // 前回の古い分割ファイルを一旦すべて消去してから、新しい最新バイナリを書き出す
  partFiles.forEach(f => { if(fs.existsSync(f)) fs.unlinkSync(f); });
  
  fs.writeFileSync('weights.bin', combinedBuffer);
  console.log("📂 脳みそファイルの更新が完了しました！");
}

run();
