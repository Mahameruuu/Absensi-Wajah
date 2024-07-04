import cv2
import os
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from datetime import datetime

# Fungsi untuk menampilkan pesan selesai di GUI
def show_message(message):
    instructions.config(text=message)

# Fungsi untuk merekam data wajah
def record_face_data():
    wajahDir = 'datawajah'
    if not os.path.exists(wajahDir):
        os.makedirs(wajahDir)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Menggunakan backend DirectShow
    if not cam.isOpened():
        show_message("Error: Kamera tidak tersedia")
        return

    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
    faceID = entry2.get()
    nama = entry1.get()
    nim = entry2.get()
    kelas = entry3.get()
    ambilData = 1

    while True:
        retV, frame = cam.read()
        if not retV:
            show_message("Error: Gagal membaca frame dari kamera")
            break

        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            namaFile = f"{nim}_{nama}_{kelas}_{ambilData}.jpg"
            cv2.imwrite(os.path.join(wajahDir, namaFile), abuabu[y:y + h, x:x + w])
            ambilData += 1

            roiabuabu = abuabu[y:y + h, x:x + w]
            roiwarna = frame[y:y + h, x:x + w]

            eyes = eyeDetector.detectMultiScale(roiabuabu)
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(roiwarna, (xe, ye), (xe + we, ye + he), (0, 255, 255), 1)

        cv2.imshow('webcamku', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Jika menekan tombol q, hentikan rekaman
            break
        elif ambilData > 30:
            break

    show_message("Rekam Data Telah Selesai!")
    cam.release()
    cv2.destroyAllWindows()

# Fungsi untuk melatih model pengenalan wajah
def train_face_recognition():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'

    if not os.path.exists(latihDir):
        os.makedirs(latihDir)

    def get_image_label(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        faceIDs = []

        for imagePath in imagePaths:
            PILimg = Image.open(imagePath).convert('L')
            imgNum = np.array(PILimg, 'uint8')
            faceID = int(os.path.split(imagePath)[-1].split('_')[0])
            faces = faceDetector.detectMultiScale(imgNum)

            for (x, y, w, h) in faces:
                faceSamples.append(imgNum[y:y + h, x:x + w])
                faceIDs.append(faceID)

        return faceSamples, faceIDs

    # Inisialisasi LBPH Face Recognizer
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces, IDs = get_image_label(wajahDir)
    faceRecognizer.train(faces, np.array(IDs))

    # Simpan model yang telah dilatih
    faceRecognizer.save(os.path.join(latihDir, 'training.yml'))
    show_message("Training Wajah Telah Selesai!")

# Fungsi untuk menandai kehadiran berdasarkan pengenalan wajah
def mark_attendance(name):
    try:
        with open("Attendance.csv", 'a+') as f:
            f.seek(0)
            namesDataList = f.readlines()
            nameList = [line.split(',')[0] for line in namesDataList]

            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.write(f'{name},{entry3.get()},{entry2.get()},{dtString}\n')

    except PermissionError as e:
        show_message(f"Permission Error: {e}")

    except Exception as e:
        show_message(f"Error: {e}")

# Fungsi untuk melakukan absensi wajah secara otomatis
def automatic_face_attendance():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        show_message("Error: Kamera tidak tersedia")
        return

    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(os.path.join(latihDir, 'training.yml'))
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = None
    yourname = entry1.get()
    names = []
    names.append(yourname)
    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)

    while True:
        retV, frame = cam.read()
        if not retV:
            show_message("Error: Gagal membaca frame dari kamera")
            break

        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeight)))

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = faceRecognizer.predict(abuabu[y:y + h, x:x + w])

            if (confidence < 100):
                id = names[0]
                confidence = "  {0}%".format(round(150 - confidence))
            elif confidence < 50:
                id = names[0]
                confidence = "  {0}%".format(round(170 - confidence))
            elif confidence > 70:
                id = "Tidak Diketahui"
                confidence = "  {0}%".format(round(150 - confidence))

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('ABSENSI WAJAH', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Jika menekan tombol q, hentikan absensi
            break

    if id is not None:  # Jika id berhasil dideteksi
        mark_attendance(id)
        show_message("Absensi Telah Dilakukan")

    cam.release()
    cv2.destroyAllWindows()

# Fungsi untuk melakukan login dengan pengenalan wajah
def login():
    latihDir = 'latihwajah'

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        show_message("Error: Kamera tidak tersedia")
        return

    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(os.path.join(latihDir, 'training.yml'))
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = None
    yourname = entry1.get()
    names = []
    names.append(yourname)
    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)

    while True:
        retV, frame = cam.read()
        if not retV:
            show_message("Error: Gagal membaca frame dari kamera")
            break

        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeight)))

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = faceRecognizer.predict(abuabu[y:y + h, x:x + w])

            if (confidence < 100):
                id = names[0]
                confidence = "  {0}%".format(round(150 - confidence))
            elif confidence < 50:
                id = names[0]
                confidence = "  {0}%".format(round(170 - confidence))
            elif confidence > 70:
                id = "Tidak Diketahui"
                confidence = "  {0}%".format(round(150 - confidence))

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Login - Pengenalan Wajah', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Jika menekan tombol q, hentikan proses login
            break

    if id is not None:  # Jika id berhasil dideteksi
        mark_attendance(id)
        show_message("Login Berhasil")

    cam.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Face Attendance - Smart Absensi")

# Mengatur canvas (window tkinter)
canvas = tk.Canvas(root, width=700, height=400)
canvas.grid(columnspan=3, rowspan=8)
canvas.configure(bg="black")

# Judul
judul = tk.Label(root, text="Face Attendance - Smart Absensi", font=("Roboto",34),bg="#242526", fg="white")
canvas.create_window(350, 80, window=judul)

# Credit
made = tk.Label(root, text="Absensi Wajah Menggunakan Python", font=("Times New Roman",13), bg="black",fg="white")
canvas.create_window(360, 20, window=made)

# Entry fields untuk data siswa
entry1 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 170, height=25, width=411, window=entry1)
label1 = tk.Label(root, text="Nama Siswa", font="Roboto", fg="white", bg="black")
canvas.create_window(90,170, window=label1)

entry2 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 210, height=25, width=411, window=entry2)
label2 = tk.Label(root, text="NIM", font="Roboto", fg="white", bg="black")
canvas.create_window(60, 210, window=label2)

entry3 = tk.Entry(root, font="Roboto")
canvas.create_window(457, 250, height=25, width=411, window=entry3)
label3 = tk.Label(root, text="Kelas", font="Roboto", fg="white", bg="black")
canvas.create_window(65, 250, window=label3)

# Global instructions label
instructions = tk.Label(root, text="Welcome", font=("Roboto",15),fg="white",bg="black")
canvas.create_window(370, 300, window=instructions)

# Tombol untuk rekam data wajah
rekam_btn = tk.Button(root, text="Ambil Gambar", font="Roboto", bg="#20bebe", fg="white", height=1, width=15, command=record_face_data)
rekam_btn.grid(column=0, row=7)

# Tombol untuk training wajah
training_btn = tk.Button(root, text="Training", font="Roboto", bg="#20bebe", fg="white", height=1, width=15, command=train_face_recognition)
training_btn.grid(column=1, row=7)

# Tombol untuk absensi wajah otomatis
absensi_btn = tk.Button(root, text="Absensi Otomatis", font="Roboto", bg="#20bebe", fg="white", height=1, width=20, command=automatic_face_attendance)
absensi_btn.grid(column=2, row=7)

# Tombol untuk login menggunakan pengenalan wajah
login_btn = tk.Button(root, text="Login", font="Roboto", bg="#20bebe", fg="white", height=1, width=15, command=login)
login_btn.grid(column=1, row=8)

root.mainloop()
