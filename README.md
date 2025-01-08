# Implementasi dan Analisis Perbandingan Efisiensi Algoritma Dijkstra dan Depth-First Search dalam Optimasi Model Lotka-Volterra untuk Dinamika Ekosistem

Proyek ini bertujuan untuk mengimplementasikan dan menganalisis perbandingan efisiensi Algoritma Dijkstra dan Depth-First Search (DFS) dalam mengoptimalkan parameter model Lotka-Volterra. Model ini digunakan untuk menggambarkan dinamika interaksi antara predator dan mangsa dalam suatu ekosistem.

## Deskripsi
Implementasi dilakukan dalam bahasa Python untuk menganalisis dan membandingkan kedua algoritma berdasarkan:

- **Kompleksitas waktu**
- **Akurasi hasil**

## Struktur Proyek

- `implementasi.py` - Kode sumber implementasi program.
- `Implementasi dan Analisis Perbandingan Efisiensi Algoritma Dijkstra dan Depth-First Search dalam Optimasi Model Lotka-Volterra untuk Dinamika Ekosistem.pdf` - Dokumen yang berisi penjelasan teori, metodologi, dan hasil analisis.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- NetworkX
- SciPy

## Fitur
Program ini mencakup implementasi berikut:

- Pembangkit dataset fiktif berbasis model Lotka-Volterra.
- Algoritma Dijkstra untuk optimasi parameter berbasis graf berbobot.
- Algoritma DFS untuk eksplorasi graf parameter.
- Perhitungan error relatif untuk menilai akurasi hasil optimasi.
- Visualisasi data simulasi dan perbandingan performa algoritma.

## Penggunaan
Untuk menjalankan program:

1. Pastikan semua dependensi telah diinstal.
2. Jalankan file `implementasi.py` dengan Python.
3. Program akan melakukan langkah-langkah berikut:
   - Membuat dataset fiktif berbasis model Lotka-Volterra.
   - Melakukan optimasi parameter menggunakan algoritma Dijkstra dan DFS.
   - Menampilkan hasil berupa visualisasi dan analisis perbandingan kedua algoritma.

## Hasil
Berdasarkan pengujian:

- Algoritma Dijkstra lebih akurat dalam menemukan jalur optimal pada graf berbobot, meskipun membutuhkan waktu lebih lama.
- Algoritma DFS lebih cepat dalam eksplorasi graf sederhana, tetapi tingkat akurasinya sedikit lebih rendah dibandingkan Dijkstra.
- Pada dataset besar dengan \(n = 5000\) dan kompleksitas graf tinggi, DFS menyelesaikan iterasi 5 kali lebih cepat dibandingkan Dijkstra.

## Author
**Muh. Rusmin Nurwadin (13523068)**  
Program Studi Teknik Informatika  
Institut Teknologi Bandung

## Referensi
Referensi lengkap dapat ditemukan pada dokumen makalah.
