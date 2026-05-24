# MSML-Dicoding_Hana-Fithri-Sabiila

##Deskripsi Proyek
Proyek ini merupakan implementasi sistem machine learning untuk memprediksi kemungkinan seorang karyawan mengalami attrition (keluar dari perusahaan). Model dibangun menggunakan algoritma Random Forest Classifier dengan pendekatan supervised learning. Tujuan utama proyek ini adalah membantu perusahaan dalam menganalisis faktor-faktor yang memengaruhi tingkat attrition sehingga perusahaan dapat mengambil langkah preventif untuk meningkatkan retensi karyawan.

# Dataset

Dataset yang digunakan adalah **Employee Attrition Dataset**. Terdiri dari 10.000 data.

## Informasi Dataset
Dataset berisi data terkait kondisi dan aktivitas karyawan dalam perusahaan, seperti:
- Usia karyawan
- Gender
- Department
- Job Role
- Pendapatan bulanan
- Jam kerja rata-rata per minggu
- Work life balance
- Job satisfaction
- Overtime

### Distribusi Target
- No  → Karyawan tetap bekerja
- Yes → Karyawan keluar dari perusahaan

---

# Tujuan Model

Model machine learning ini digunakan untuk:

- Memprediksi kemungkinan attrition karyawan
- Membantu HR melakukan analisis risiko resign
- Menentukan faktor yang paling memengaruhi attrition
- Mendukung pengambilan keputusan berbasis data

---

# Algoritma yang Digunakan

## Random Forest Classifier
Alasan menggunakan Random Forest:
- Cocok untuk data klasifikasi
- Mampu menangani banyak fitur
- Memiliki performa yang stabil
- Mengurangi risiko overfitting dibanding decision tree biasa
- Dapat digunakan untuk melihat feature importance

Library utama:
- pandas
- scikit-learn
- mlflow

---

#Hasil Proyek
Model berhasil digunakan untuk melakukan prediksi attrition karyawan berdasarkan data HR dan aktivitas kerja.
Selain model machine learning, proyek ini juga telah dilengkapi dengan:
- Experiment tracking menggunakan MLflow
- Deployment API
- Monitoring Prometheus
- Dashboard Grafana
- Alerting system
- Docker containerization

| Kolom | Keterangan |
|---|---|
| Attrition | Status apakah karyawan keluar dari perusahaan atau tidak |
