async function uploadAndTrain() {
    const fileInput = document.getElementById('fileInput');
    const statusTxt = document.getElementById('trainStatus');
    
    if(fileInput.files.length === 0) {
        alert("Pilih file Excel dulu!");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    statusTxt.innerText = "⏳ Melatih NB, SVM, & Logistic Regression...";
    statusTxt.className = "mt-3 text-sm text-center text-blue-600 animate-pulse";

    try {
        const response = await fetch('/api/upload_train', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if(data.status === 'success') {
            statusTxt.innerText = "✅ Training Selesai!";
            statusTxt.className = "mt-3 text-sm text-center text-green-600 font-bold";
            
            // 1. UPDATE TABEL EVALUASI
            const tableBody = document.getElementById('evalTableBody');
            tableBody.innerHTML = ""; // Bersihkan isi lama
            
            // Cari skor tertinggi untuk highlight
            let maxAcc = 0;
            const models = data.evaluation;
            
            // Loop data metrics
            for (const [modelName, metrics] of Object.entries(models)) {
                if (metrics.accuracy > maxAcc) maxAcc = metrics.accuracy;
            }

            for (const [modelName, metrics] of Object.entries(models)) {
                const isBest = metrics.accuracy === maxAcc;
                const rowClass = isBest ? "bg-green-50 font-semibold" : "border-b";
                const badge = isBest ? "🥇" : "";

                const row = `
                    <tr class="${rowClass}">
                        <td class="px-3 py-3 text-slate-800">${modelName} ${badge}</td>
                        <td class="px-3 py-3 text-green-600">${metrics.accuracy}%</td>
                        <td class="px-3 py-3 text-blue-600">${metrics.f1_score}%</td>
                    </tr>
                `;
                tableBody.innerHTML += row;
            }
            document.getElementById('modelStats').classList.remove('hidden');

            // 2. UPDATE WORDCLOUDS (Pisah Positif & Negatif)
            document.getElementById('wcContainer').classList.remove('hidden');
            
            if(data.wordcloud.positive) {
                document.getElementById('wcPos').src = "data:image/png;base64," + data.wordcloud.positive;
            }
            if(data.wordcloud.negative) {
                document.getElementById('wcNeg').src = "data:image/png;base64," + data.wordcloud.negative;
            }

        } else {
            statusTxt.innerText = "❌ Error: " + data.message;
            statusTxt.className = "mt-3 text-sm text-center text-red-600";
        }
    } catch (error) {
        console.error(error);
        statusTxt.innerText = "❌ Terjadi kesalahan koneksi.";
    }
}

async function predictSentiment() {
    const text = document.getElementById('predictInput').value;
    const modelChoice = document.getElementById('modelSelect').value; // Ambil pilihan model

    if(!text) return;

    const resArea = document.getElementById('resultArea');
    const resBox = document.getElementById('resultBox');
    const resLabel = document.getElementById('resultLabel');
    const resConf = document.getElementById('resultConf');
    const resIcon = document.getElementById('resultIcon');
    const resModel = document.getElementById('resultModel');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                text: text, 
                model: modelChoice // Kirim pilihan model ke backend
            })
        });
        const data = await response.json();

        if(data.status === 'success') {
            resArea.classList.remove('hidden');
            const sent = data.result.sentiment;
            
            resLabel.innerText = sent.toUpperCase();
            resConf.innerText = data.result.confidence.toFixed(2) + "%";
            resModel.innerText = data.result.model; // Tampilkan model yang dipakai

            if(sent === 'Positif') {
                resBox.className = "flex items-center gap-4 p-4 rounded-lg border bg-green-50 border-green-200 text-green-800 transition-all";
                resIcon.innerText = "😊";
            } else {
                resBox.className = "flex items-center gap-4 p-4 rounded-lg border bg-red-50 border-red-200 text-red-800 transition-all";
                resIcon.innerText = "😡";
            }
        } else {
            alert(data.message);
        }
    } catch (error) {
        console.error(error);
        alert("Error prediksi");
    }
}