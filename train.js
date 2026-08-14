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
const LEARNING_RATE = 0.5; // ガツンと学習させて収束を早める

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
  console.log("🚀 超高速バイナリ対応・自動学習エンジンを起動します。");
  console.log("🧠 6,700万個の脳みそデータ（約268MB）をメモリに展開中...");
  
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  console.log("⏳ 特訓を開始します。GitHubのサーバーで並列計算中...");

  // 200回特訓
  for (let epoch = 1; epoch <= 200; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      // 手動成功版と完全に同じ、インデックス位置に1.0をセットするアルゴリズム
      const hInputVector = new Float32Array(DIMENSIONS);
      if (inputIds[0] !== undefined) hInputVector[0] = 1.0; // 「私」
      if (inputIds[1] !== undefined) hInputVector[1] = 1.0; // 「好き」
      if (inputIds[2] !== undefined) hInputVector[2] = 1.0; // 「猫」または「犬」

      for (const correctWord of data.target) {
        const correctId = VOCAB.indexOf(correctWord);

        // 予測（Forward）：手動成功版と100%同じインデックス計算
        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          for (let i = 0; i < DIMENSIONS; i++) {
            score += hInputVector[i] * hWeightsMatrix[rowOffset + i];
          }
          resultScores[wordId] = score;
        }

        const hProbs = softmax(resultScores);

        // 誤差修正（Backward）：アルゴリズムのねじれを完全に修正
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let error = hProbs[wordId];
          if (wordId === correctId) {
            error -= 1.0;
          }
          
          if (error !== 0) {
            const rowOffset = wordId * DIMENSIONS;
            for (let i = 0; i < DIMENSIONS; i++) {
              hWeightsMatrix[rowOffset + i] -= LEARNING_RATE * error * hInputVector[i];
            }
          }
        }
      }
    }
  }

  console.log("\n✅ 特訓が完了しました！学習成果をテストします...");

  // テスト（手動成功版と100%同じ、シンプルな一発総当たりアルゴリズム）
  function testTranslation(testInputText) {
    const testIds = tokenize(testInputText);
    const hInputVector = new Float32Array(DIMENSIONS);
    // テスト文脈に合わせてインデックスをON
    if (testInputText.includes("私")) hInputVector[0] = 1.0;
    if (testInputText.includes("好き")) hInputVector[1] = 1.0;
    if (testInputText.includes("猫")) hInputVector[2] = 1.0;
    if (testInputText.includes("犬")) hInputVector[2] = 1.0; // 猫と同じ目的語スロット

    const scores = new Float32Array(VOCAB_SIZE);
    for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
      let score = 0;
      const rowOffset = wordId * DIMENSIONS;
      for (let i = 0; i < DIMENSIONS; i++) { score += hInputVector[i] * hWeightsMatrix[rowOffset + i]; }
      scores[wordId] = score;
    }
    
    const probs = softmax(scores);
    const wordProbs = [];
    for(let i=0; i<VOCAB_SIZE; i++) { if(probs[i] > (1 / VOCAB_SIZE) && VOCAB[i]) wordProbs.push({ word: VOCAB[i], prob: probs[i] }); }
    wordProbs.sort((a,b) => b.prob - a.prob);

    const result = wordProbs.slice(0, 4).map(wp => wp.word).join(" ");
    console.log(`入力: 「${testInputText}」 ➔ 🤖 翻訳結果: [ ${result} ]`);
  }

  await testTranslation("私 好き 猫");
  await testTranslation("私 好き 犬");

  // =========================================================================
  // 6. 📦 【超高速化】268MBのデータを一瞬で生のバイナリ（weights.bin）に書き出す
  // =========================================================================
  console.log("\n💾 賢くなった脳みそを『weights.bin』として爆速書き出し中...");
  
  // テキスト変換を完全にパスし、生の数字バッファをそのままファイル化（1秒未満）
  const buffer = Buffer.from(hWeightsMatrix.buffer);
  fs.writeFileSync('weights.bin', buffer);
  
  console.log("📂 バイナリファイルの保存が完了しました！");
}

run();
