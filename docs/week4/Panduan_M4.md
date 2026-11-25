# Minggu 4 — Artikel Ilmiah dan Reproduksibilitas

Minggu 4 merupakan fase konsolidasi akhir dari rangkaian eksperimen klasifikasi sinyal suara mesin FordA (M1–M3). Pada tahap ini, seluruh hasil eksperimen, perbandingan performa, uji signifikansi statistik, dan analisis efisiensi komputasi disusun ke dalam bentuk artikel ilmiah standar IMRaD serta paket materi presentasi.

---

## 1. Luaran Wajib Minggu 4

Folder ini memuat luaran utama tahap M4:
1. **Naskah Artikel Ilmiah (PDF):** `Artikel M4 — Klasifikasi Sinyal Mesin FordA.pdf`
   - Format IEEE 2-kolom ringkas (6 halaman).
   - Memuat: Abstrak, Pendahuluan, Metode (Dataset, Baseline, Pembanding, Protokol Evaluasi), Hasil (Replikasi M1, Split Asli M2, 5-Fold M3, Uji Signifikansi Holm & McNemar, Analisis Ablasi, Latensi Ekstraksi), Pembahasan Trade-off, serta Simpulan.
2. **Naskah Artikel Format Word (DOCX):** `Artikel M4 — Klasifikasi Sinyal Mesin FordA.docx`
   - Berkas naskah terformat lengkap untuk keperluan revisi dan pengarsipan akademik.
3. **Slide Presentasi Lengkap (PPTX):** `Minggu_4_Artikel_dan_Reproduksibilitas.pptx`
   - Berisi 14 slide komprehensif mencakup latar belakang, metodologi, hasil evaluasi M1–M3, dan secara khusus menjawab checklist evaluasi:
     - Flowchart metodologi end-to-end
     - Perbedaan hardware & requirements (paper vs replikasi)
     - Konfigurasi lengkap hiperparameter
     - Cara kerja & mekanisme pembelajaran QUANT vs ExtraTrees
     - Analisis kurva pembelajaran (*learning curve*)
     - Repositori GitHub dan akses artefak
4. **Skrip Eksperimen Tambahan M4:**
   - `ablation_features.py`: Eksperimen ablasi kontribusi sub-kelompok fitur Welch (statistik bentuk vs PSD spektral vs bandpower).
   - `benchmark_extraction_latency.py`: Pengukuran latensi komputasi ekstraksi fitur cold-start per sampel (QUANT vs Welch 48D).
   - `dotplot_multifold_f1.png`: Visualisasi persebaran metrik F1 lintas lima fold.

---

## 2. Ketersediaan Kode & Repositori GitHub

Seluruh kode sumber, notebook interaktif, log eksekusi, dan artefak model dipublikasikan secara terbuka pada repositori GitHub resmi:

- **URL Repositori:** [https://github.com/RafaelDaisukee/quant-forda-replication.git](https://github.com/RafaelDaisukee/quant-forda-replication.git)
- **Struktur Repositori:**
  - `notebooks/`: 
    - `01_replikasi_QUANT_FordA_selesai_30_resample.ipynb` (M1)
    - `02_pembanding_QUANT_FordA.ipynb` (M2)
    - `03_hasil_analisis.ipynb` (M3)
  - `config/`: Berkas `requirements.txt`, `requirements_actual.txt`, dan konfigurasi JSON.
  - `data/`: Skrip otomasi pengunduhan dataset FordA beserta pemeriksaan checksum SHA-256.
  - `docs/`: Slide presentasi mingguan (M1–M4) dan dokumen spesifikasi teknis.
  - `outputs/`: Tabel metrik CSV, matriks konfusi, kurva ROC, dan kurva pembelajaran.
  - `logs/`: Riwayat log terminal eksekusi dari Google Colab.

---

## 3. Ringkasan Temuan Utama

1. **Replikasi Baseline (M1):** 30 resample QUANT menghasilkan rata-rata akurasi 96,81% ± 0,46%, sangat presisi mereplikasi angka acuan penulis (96,90% ± 0,53%, selisih hanya −0,09 pp).
2. **Pembanding Fitur Welch 48D (M2 & M3):** Representasi kompak 48 fitur statistik-spektral Welch menghasilkan performa yang setara secara statistik dengan 4.469 fitur QUANT (mean Macro-F1 96,91% vs 96,79%, $p_{\text{Holm}} = 0,748$, $p_{\text{McNemar}} = 0,685$).
3. **Efisiensi Komputasi:**
   - Ekstraksi fitur Welch 48D **8,4 kali lebih cepat** dibanding QUANT (0,055 ms vs 0,463 ms per sampel).
   - Waktu pelatihan (*fit latency*) ExtraTrees pada fitur Welch 48D **46 kali lebih cepat** dibanding QUANT (0,84 detik vs 39,16 detik).
   - Memangkas dimensi fitur sebesar **98,93%** (dari 4.469 menjadi 48 fitur).
4. **Trade-off:** Variabilitas performa antarfold pada Welch 48D lebih lebar (SD 0,64 pp vs 0,21 pp), serta terdapat sedikit peningkatan False Negative akibat hilangnya lokalisasi waktu pada perataan daya spektral global.
