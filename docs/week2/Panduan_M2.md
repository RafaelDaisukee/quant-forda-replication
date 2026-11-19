# Minggu 2 — Metode pembanding QUANT dan FordA

Baseline mempertahankan kode transform QUANT asli, 4.469 fitur, dan ExtraTrees 200 pohon. Dua pembanding ditetapkan sebelum skor test M2 diperiksa:

1. **Statistik dan Welch 48D + ExtraTrees.** Mengubah representasi dengan classifier yang sama untuk menilai ringkasan global yang lebih ringkas: 8 statistik bentuk, 8 statistik PSD, dan 32 bandpower relatif. Kehilangan informasi lokasi merupakan batasan.
2. **QUANT + StandardScaler + RidgeClassifierCV.** Mengubah classifier dengan representasi yang sama. Regularisasi L2 membatasi koefisien pada fitur berkorelasi. Alpha dipilih lewat efficient leave-one-out pada train, dari 10 nilai 10^-3 sampai 10^3, dengan kriteria bawaan negative MSE. Paper membahas Ridge pada Bagian 4.2.4/Gambar 23; implementasi ini tidak diklaim sebagai salinan persis kode ablasinya.

Semua metode memakai 3.601 train dan 1.320 test lengkap dengan indeks asli yang sama. Label −1/+1 dipertahankan. Frekuensi Welch dinormalisasi dalam cycles/sample; sample rate fisik tidak diasumsikan.

Notebook: `../output/jupyter-notebook/02_pembanding_QUANT_FordA.ipynb`. Jalankan dari atas pada runtime CPU Colab. Dataset diverifikasi dengan checksum; cache M1 dipakai jika cocok. Jika cache tidak tersedia, transform dihitung ulang dari kode penulis. Hasil tersimpan terpisah di `dsp_quant_forda/runs/m2_original_split`.

Run lokal dari folder ini:

```text
python run_comparison.py --root ../m1_quant_forda/data --out results_new --m1-features ../m1_quant_forda/results_local_cpu/features.npy
```

Tanpa `--m1-features`, fitur dihitung sendiri. Dependensi mengikuti M1 dengan PyTorch CPU. `results_local` menyimpan uji implementasi Windows CPU, bukan hasil Colab atau lima fold: accuracy baseline 96,8182%, Welch48 97,3485%, dan Ridge 95,7576%. Hasil satu split belum membuktikan peningkatan signifikan. Waktu membaca cache bukan waktu ekstraksi baru.

Learning curve menggunakan 720/1.440/2.160/2.880 kasus train dan 721 validasi dari train utama. Test tidak dipakai untuk kurva atau pemilihan parameter. Evaluasi mencakup accuracy, balanced accuracy (Macro-Accuracy), Macro-F1, AUC, serta waktu. AUC Ridge memakai decision_function.

Artefak: konfigurasi, indeks split, model, prediksi, tabel metrik, confusion matrix, ROC, learning curve, lingkungan, dan requirements. Tiga pemeriksaan implementasi serta rerun cache telah lulus.

M3 belum dijalankan. Rencana: lima Stratified Fold bersama pada 4.921 kasus, scaler hanya fit pada train fold, paired t-test dua sisi dan Holm untuk dua pembanding terhadap baseline pada Macro-F1. Laporkan keterbatasan lima fold dan hasil yang tidak signifikan.
