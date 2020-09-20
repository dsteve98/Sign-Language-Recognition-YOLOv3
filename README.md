# Sign-Language-Recognition-YOLOv3

Google Colab:
- listmaker : untuk membuat train.txt dan test.txt (list gambar untuk training dan testing)
- Run-Train : untuk training, jalankan bagian "first installation" jika belum ada darknet pada GDrive
- Image-Test : untuk testing gambar,output hasil deteksi bentuk .txt
- Image-Test-confusion matrix : untuk evaluasi hasil, parsing dari .txt hasil Image-Test

Local:
- Dites pada Ubuntu 16.04, compile darknet seperti di colab dengan LIBSO=1.
- requirement : https://github.com/AlexeyAB/darknet#requirements
- video-test.py : untuk testing video. nilai webcam tergantung perangkat (bisa -1,0,1,dll)

Both: sesuaikan dengan lokasi pada local/gdrive pada kode
