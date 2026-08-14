const fs = require('fs');

// =========================================================================
// 1. 共通辞書とロマンスペックの設定
// =========================================================================
// ここに単語を何個足しても、下の計算コードは1文字もいじる必要はありません
const VOCAB = [
  "<PAD>", "<BOS>", "<EOS>", "<UNK>", // 0, 1, 2, 3
  "私", "猫", "好き", "犬",            // 4, 5, 6, 7 (日本語)
  "I", "cats", "like", ".", "dogs"     // 8, 9, 10, 11, 12 (英語)
];
const VOCAB_SIZE = 65536;  
const DIMENSIONS = 1024;  
const WEIGHTS_COUNT = VOCAB_SIZE * DIMENSIONS;
const LEARNING_RATE = 0.5; 

// =========================================================================
// 2. 訓練データ（ここに好きな文章を足すだけで自動で部屋が分かれます）
// =========================================================================
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

async function run() {
  console.log("🚀 真・全自動拡張型AIエンジンを起動します。");
  console.log("🧠 6,700万個の脳みそデータ（約268MB）をメモリに展開中...");
  
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  console.log("⏳ 特訓を開始します。部屋の自動割り当てを実行中...");

  // 200回特訓
  for (let epoch = 1; epoch <= 200; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      // ⭕【真の自動化アルゴリズム】
      // 単語のID番号をそのまま1024次元のメーターの位置としてONにする（これで完全な拡張性を確保！）
      const hInputVector = new Float32Array(DIMENSIONS);
      inputIds.forEach(id => {
        if (id < DIMENSIONS) {
          hInputVector[id] = 1.0; 
        }
      });

      for (const correctWord of data.target) {
        const correctId = VOCAB.indexOf(correctWord);

        // 予測（Forward）：自動で分かれた部屋を使ってスコアを計算
        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          // ONになっている単語の部屋の重みだけをピンポイントで足し算する
          inputIds.forEach(id => {
            if (id < DIMENSIONS) {
              score += hInputVector[id] * hWeightsMatrix[rowOffset + id];
            }
          });
          resultScores[wordId] = score;
        }

        const hProbs = softmax(resultScores);

        // 誤差修正（Backward）：自動で分かれた部屋の重みだけを正確に修正
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let error = hProbs[wordId];
          if (wordId === correctId) {
            error -= 1.0;
          }
          
          if (error !== 0) {
            const rowOffset = wordId * DIMENSIONS;
            // 入力された単語の部屋（次元）の重みだけを自力修正する
            inputIds.forEach(id => {
              if (id < DIMENSIONS) {
                hWeightsMatrix[rowOffset + id] -= LEARNING_RATE * error * hInputVector[id];
              }
            });
          }
        }
      }
    }
  }

  console.log("\n✅ 特訓が完了しました！学習成果をテストします...");

  // テスト（完全に自動化された一発総当たりアルゴリズム）
  function testTranslation(testInputText) {
    const testIds = tokenize(testInputText);
    const hInputVector = new Float32Array(DIMENSIONS);
    testIds.forEach(id => {
      if (id < DIMENSIONS) hInputVector[id] = 1.0;
    });

    const scores = new Float32Array(VOCAB_SIZE);
    for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
      let score = 0;
      const rowOffset = wordId * DIMENSIONS;
      testIds.forEach(id => {
        if (id < DIMENSIONS) score += hInputVector[id] * hWeightsMatrix[rowOffset + id];
      });
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

  console.log("\n💾 賢くなった脳みそを『weights.bin』として爆速書き出し中...");
  const buffer = Buffer.from(hWeightsMatrix.buffer);
  fs.writeFileSync('weights.bin', buffer);
  console.log("📂 バイナリファイルの保存が完了しました！");
}

run();
