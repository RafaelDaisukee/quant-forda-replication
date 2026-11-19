# Minggu 3 — Hasil dan analisis FordA

Notebook `../output/jupyter-notebook/03_hasil_analisis.ipynb` menjalankan tiga metode M2 pada lima fold yang sama. Pilih runtime CPU Colab dan jalankan dari atas. Dataset lengkap dan cache M1 diperiksa; hasil disimpan terpisah di `dsp_quant_forda/runs/m3_stratified5fold_seed42`.

## Protokol

Seluruh 4.921 kasus FordA digabung untuk StratifiedKFold 5, shuffle=True, seed 42. Ini analisis CV eksploratif dengan protokol berbeda dari split resmi M2 dan 30 resample M1. Data test resmi dapat masuk train fold M3. Jangan menyebutnya konfirmasi eksternal atau membandingkan angkanya langsung dengan protokol sebelumnya.

Metode tetap: QUANT asli + ExtraTrees 200; statistik–Welch48 + ExtraTrees yang sama; QUANT + StandardScaler + RidgeClassifierCV. Classifier seed 0. Scaler hanya fit pada train fold; alpha dipilih di train fold. Tidak ada tuning berdasarkan hasil fold test.

## Analisis

- Tabel 15 hasil fold dan mean ± SD sampel untuk accuracy, Macro-Accuracy (balanced accuracy), Macro-F1, dan AUC.
- Paired t-test dua sisi untuk Macro-F1, pembanding minus baseline, dengan Holm untuk dua pembanding. Laporkan hasil tidak signifikan atau arah penurunan jika memang itu hasilnya. Lima fold berdaya rendah dan training saling tumpang tindih; koreksi Holm tidak mengatasi ketergantungan fold.
- Confusion matrix out-of-fold mencakup setiap kasus tepat sekali. ROC dihitung per fold. Pooled AUC Ridge tidak dilaporkan karena skala skor model fold dapat berbeda.
- Boxplot metrik dan F1 per kelas. Learning curve diagnostik di dalam train fold pertama, dengan 20% validasi internal dan empat ukuran train. Test fold pertama tidak dipakai dan kurva tidak dipakai untuk memilih parameter.
- Semua FP/FN disimpan; contoh deterministik dari Welch48 disertai sinyal dan PSD. +1 merupakan kelas positif untuk perhitungan, tanpa asumsi jenis kerusakan fisik.

## Cache dan menjalankan lokal

Setiap pekerjaan menyimpan model, prediksi, metrik, dan checksum secara terpisah. Pekerjaan selesai yang valid dilewati pada rerun. Cache berubah/rusak menghentikan proses agar eksperimen tidak tercampur. Jika cache fitur M1 tidak cocok, set `M1_FEATURES=None` untuk menghitung fitur ulang. Gunakan folder RUN baru jika sengaja mengubah kode, konfigurasi, atau lingkungan.

```text
python run_analysis.py --root ../m1_quant_forda/data --out results_new --m1-features ../m1_quant_forda/results_local_cpu/features.npy
python test_analysis.py
```

Dependensi mengikuti M1/M2: Python yang kompatibel dengan NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.6.1, PyTorch 2.6.0+cpu, dan paket pada requirements notebook. `results_local_cpu` adalah validasi lokal Windows, bukan hasil Colab.

Seluruh 15 model utama dan 12 titik kurva diperiksa dalam run lengkap lokal. Lima pemeriksaan mencakup fold, pairing/statistik, Holm, kelengkapan OOF, serta deteksi cache korup. Notebook mengandung sumber yang sama dengan program yang diuji. Hasil Colab baru dapat disebut selesai setelah notebook dijalankan dan outputnya diperiksa.

Spesifikasi sebelum eksperimen: `SPEC.md`. Sumber: [StratifiedKFold](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.StratifiedKFold.html), [paired t-test](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.ttest_rel.html), [paper QUANT](https://doi.org/10.1007/s10618-024-01036-9).
