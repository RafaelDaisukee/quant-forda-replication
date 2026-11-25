# Engine Sound Classification pada FordA

Replikasi QUANT dan perbandingan fitur statistik–spektral Welch serta classifier Ridge. Artikel M4 berformat dua kolom gaya IEEE, 4 halaman sebelum identitas penulis ditambahkan.

## Sumber dan cakupan

- Paper utama: Dempster, Schmidt, Webb (2024), [quant: a minimalist interval method for time series classification](https://doi.org/10.1007/s10618-024-01036-9).
- [Kode dan hasil penulis](https://github.com/angus924/quant). Commit transform: `5956b7252464d2c17e3b541e051b05f03c7442ff`; commit hasil acuan: `f3907f7e531b6a304e3c54310869a0023a9e0fd6`.
- [Dataset FordA](https://www.timeseriesclassification.com/description.php?Dataset=FordA): 3.601 train, 1.320 test, panjang 500, label −1/+1. Seluruh data dipakai. Tidak ada asumsi sample rate fisik atau pemetaan label ke komponen rusak tertentu.
- Arsip [FordA.zip](https://www.timeseriesclassification.com/aeon-toolkit/FordA.zip), SHA256 `4d78a72e5165e8da40c51115a8219c9d01978aafac6334ed8dfabd29da359ffb`.

## Progres dan protokol

| Tahap | Protokol | Hasil Colab |
|---|---|---|
| M1 | QUANT asli, 30 resample, data lengkap | Mean accuracy 96,8081% ± 0,4622 pp; acuan 96,9015% ± 0,5254 pp |
| M2 | Split asli 3.601/1.320 untuk tiga metode | Accuracy baseline 96,74%; Welch48 97,35%; Ridge 95,83% |
| M3 | Pool 4.921 kasus, shared StratifiedKFold lima fold, seed 42 | Macro-F1 96,79 ± 0,21%; 96,91 ± 0,64%; 95,75 ± 0,55% |
| M4 | Artikel IMRaD, sitasi, diskusi, paket reproduksi | `docs/Artikel_M4.pdf` dan DOCX; identitas penulis diisi pengguna |

M1 resample 19 berbeda −2,3485 pp dari acuan dan tetap disertakan. Mean yang dekat bukan bukti hasil identik. Seed/random draw dan lingkungan penulis belum seluruhnya identik; penyebab setiap selisih belum diisolasi.

M3 adalah protokol berbeda dari holdout M1/M2: test resmi dapat masuk train fold. Dataset telah dilihat pada M2, sehingga M3 bersifat eksploratif, bukan konfirmasi eksternal. Macro-Accuracy berarti balanced accuracy; SD memakai ddof=1. Metrik utama uji berpasangan adalah Macro-F1. Welch p Holm 0,748050; Ridge p mentah 0,035740 dan p Holm 0,071481. Tidak ada perbedaan signifikan setelah Holm, dan ini tidak membuktikan kesetaraan. Lima fold memiliki train set yang tumpang tindih.

Mean fit classifier M3: baseline 39,1598 s; Welch48 0,8369 s; Ridge 27,8020 s. Waktu tersebut bukan cold-start pipeline dan tidak mencakup ekstraksi baru. Metadata sumber mesin/sesi tidak tersedia untuk group-CV.

## Struktur

- `notebooks/`: tiga snapshot notebook yang telah selesai di Colab, beserta output.
- `src/`: skrip tiap tahap, kode vendor, source lock, lisensi asli.
- `results/m1`, `m2`, `m3`: bukti run Colab, konfigurasi/lingkungan, CSV, JSON, gambar, provenance.
- `docs/`: artikel PDF dan sumber Word yang dapat diedit.
- `ARTIFACTS.md`: tautan notebook aktif, snapshot, serta model/prediksi/cache pada Drive.
- `MANIFEST.json`: SHA256 berkas paket. `verify_artifacts.py` memeriksa paket dan konsistensi ringkasan tanpa training.

## Reproduksi di Colab

1. Buka notebook aktif pada ARTIFACTS.md. Pilih runtime CPU; metode tidak membutuhkan GPU atau training berbasis epoch.
2. Jalankan setup, mount Drive, dan tahap yang disetujui. Kode, data, dan model lengkap mengikuti notebook; gunakan akun Drive sendiri atau salinan cache yang dapat dibaca.
3. Dataset diunduh jika arsip/cache belum ada atau gagal validasi. Hasil dapat dipakai kembali hanya setelah identitas konfigurasi, versi paket, source lock, dan checksum yang relevan cocok. Folder run yang tidak cocok tidak ditimpa diam-diam.
4. M1 menjalankan 30 resample. M2 menjalankan tiga metode dan learning curve internal. M3 menjalankan 15 model outer-fold dan 12 model learning curve internal. Jangan mengganti hasil dengan run terbatas.
5. Hasil yang terputus dilanjutkan per job yang lengkap; job yang belum selesai dapat diulang. Persistensi Drive harus diperiksa sebelum runtime diganti. Cache bukan jaminan semua pekerjaan yang terputus tersimpan.

Lingkungan hasil utama: Linux Colab, Python 3.13.18, paket persis dalam requirements dan `results/*/environment.json`. Perbedaan perangkat/versi dapat mengubah hasil numerik; catat sebagai run baru. PyTorch dipakai untuk transform, bukan neural-network training.

Alternatif di terminal **Colab**, sesudah eksperimen disetujui:

```sh
python -m pip install -r requirements.txt
python run.py m1 --root /content/forda_data --out /content/drive/MyDrive/dsp_quant_forda/m1_new
python run.py m2 --root /content/forda_data --out /content/drive/MyDrive/dsp_quant_forda/m2_new
python run.py m3 --root /content/forda_data --out /content/drive/MyDrive/dsp_quant_forda/m3_new
```

Tanpa `--m1-features`, M2/M3 menghitung QUANT kembali dengan metode yang sama. Untuk menggunakan cache M1, sertakan `--m1-features` ke file NPY asli dengan sidecar/lingkungan/config yang sesuai; cache gagal validasi perlu diperiksa, bukan dipaksa dilewati.

Pemeriksaan paket tanpa training: `python verify_artifacts.py`. Perintah `python run.py --help` hanya menampilkan opsi. Tidak ada training lokal untuk penyusunan M4; validasi paket tidak menguji ulang performa model.

## Lisensi dan publikasi

Lisensi GPL penulis QUANT dan BSD komponen resampling disertakan dalam folder vendor masing-masing. Pertahankan atribusi dan lisensi saat membagikan kode; paket ini tidak mengubah lisensi sumber pihak ketiga. Data/model/cache besar tidak disertakan dalam Git, tersedia melalui tautan dan prosedur notebook. ZIP ini siap dipindahkan ke repositori, tetapi belum dipublikasikan ke GitHub karena belum ada remote tujuan.

[PPT M4](https://docs.google.com/presentation/d/1n1h5vZaVX8uXIMwZIl9HcplW8Bj4JpTa/edit).
