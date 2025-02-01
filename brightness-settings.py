import cv2
import mediapipe as mp
import screen_brightness_control as sbc

# Inisialisasi MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Membuka kamera
cap = cv2.VideoCapture(0)

# Membuat jendela fullscreen
cv2.namedWindow("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mengubah gambar ke format RGB untuk deteksi MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Mendapatkan posisi jari telunjuk dan tengah (landmark 8 dan 12)
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]

            # Menghitung jarak antara jari telunjuk dan tengah
            distance = ((index_finger.x - middle_finger.x)**2 + (index_finger.y - middle_finger.y)**2)**0.5

            # Mengatur kecerahan layar berdasarkan jarak tangan
            brightness = int(min(max(distance * 1000, 0), 100))  # Menyusun kecerahan antara 0-100
            sbc.set_brightness(brightness)

    # Menampilkan gambar dengan deteksi tangan dalam mode fullscreen
    cv2.imshow("Hand Gesture Recognition", frame)

    # Keluar jika menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
