// Node.js内蔵のファイル書き込みモジュールをインポート
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

// ========================================================================
// 2. 訓練データ（ここにデータを足すだけで自動学習します！）
// =========================================================================
const trainData = [
  { input: "私 好き 猫", target: ["I", "like", "cats", "."] },
  { input: "私 好き 犬", target: ["I", "like", "dogs", "."] } 
];

const tokenize = (text) => text.split(" ").map(w => VOCAB.includes(w) ? VOCAB.indexOf(w) : 3);

// 簡易ソフトマックス関数（Node.jsのCPU処理用）
function softmax(scores) {
  const maxScore = Math.max(...scores);
  let sumExp = 0;
  const exps = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) { exps[i] = Math.exp(scores[i] - maxScore); sumExp += exps[i]; }
  return exps.map(v => v / sumExp);
}

// 確定で一番良い単語を選ぶ（Argmax）
function getBestWord(probs) {
  return probs.indexOf(Math.max(...probs));
}

async function run() {
  console.log("🚀 GitHub Actions 上で自動学習エンジンが起動しました。");
  console.log("🧠 6,700万個の脳みそデータ（約268MB）をメモリに展開中...");

  // 最初は完全に「0（無知）」の脳みそ行列（268MB）を作成！
  const hWeightsMatrix = new Float32Array(WEIGHTS_COUNT);

  // =========================================================================
  // 5. 🔁 自動学習ループ（2つのデータを読み込ませて200回特訓）
  // =========================================================================
  console.log("⏳ 特訓を開始します。GitHubのサーバーが代わりにフル稼働しています...");

  for (let epoch = 1; epoch <= 200; epoch++) {
    for (const data of trainData) {
      const inputIds = tokenize(data.input);
      
      // 自動次元割当
      const hInputVector = new Float32Array(DIMENSIONS);
      inputIds.forEach(id => { if(id < DIMENSIONS) hInputVector[id] = 1.0; });

      for (const correctWord of data.target) {
        const correctId = VOCAB.indexOf(correctWord);

        // --- ステップA: 今の脳みそで予測（Forward） ---
        const resultScores = new Float32Array(VOCAB_SIZE);
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          let score = 0;
          const rowOffset = wordId * DIMENSIONS;
          for (let i = 0; i < DIMENSIONS; i++) {
            score += hInputVector[i] * hWeightsMatrix[rowOffset + i];
          }
          resultScores[wordId] = score;
        }

        // 確率（Softmax）に変換
        const hProbs = softmax(resultScores);

        // --- ステップB: 間違えたら脳みそを自動修正（Backward） ---
        const wordErrors = new Float32Array(VOCAB_SIZE);
        for (let i = 0; i < VOCAB_SIZE; i++) {
          wordErrors[i] = hProbs[i];
          if (i === correctId) wordErrors[i] -= 1.0;
        }

        // 6,700万個の重みを一斉に書き換える（勾配降下法）
        for (let wordId = 0; wordId < VOCAB_SIZE; wordId++) {
          const error = wordErrors[wordId];
          const rowOffset = wordId * DIMENSIONS;
          if (error !== 0) { // 計算を少しでも高速化するセーフティ
            for (let i = 0; i < DIMENSIONS; i++) {
              hWeightsMatrix[rowOffset + i] -= LEARNING_RATE * error * hInputVector[i];
            }
          }
        }
      }
    }
  }

  console.log("\n✅ GitHub Actions での自動学習が200回無事に完了しました！");
  console.log("🧪 学習成果をテストします...");

  // テスト関数
  function testTranslation(testInputText) {
    const testIds = tokenize(testInputText);
    const hInputVector = new Float32Array(DIMENSIONS);
    testIds.forEach(id => { if(id < DIMENSIONS) hInputVector[id] = 1.0; });

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
  // 6. 📦 賢くなった「脳みそのデータ（約268MB）」をJSONファイルとして書き出し
  // =========================================================================
  console.log("\n💾 賢くなった脳みそ（重みデータ）を『weights.json』として書き出し中...");
  
  // Float32Arrayのままだと書き出せないので、通常の配列に戻してシリアライズ
  const weightsArray = Array.from(hWeightsMatrix);
  fs.writeFileSync('weights.json', JSON.stringify(weightsArray));
  
  console.log("📂 ファイルの書き出しが成功しました。これを受け取ってGitHub Actionsが保存します！");
}

run();
