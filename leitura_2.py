import cv2
import numpy as np
import pandas as pd

# ========= CONFIG =========
video_path = "pendulo_preto.mp4"
output_csv = "pendulo_posicoes.csv"

# AJUSTE MANUAL (IMPORTANTE)
# Coloque um retângulo que contenha a massa no primeiro frame
#x0, y0, w, h = 880, 500, 120, 120  # <-- MUDE ISSO
x0, y0 = 1100, 720
w, h = 100, 100
pad = 80  # área de busca ao redor do ponto
# ==========================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Erro ao abrir o vídeo")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Total de frames: {frame_count}")

dados = []

# posição inicial
cx_prev = x0 + w // 2
cy_prev = y0 + h // 2

trajetoria = []

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]

    # Região de busca
    x1 = max(cx_prev - pad, 0)
    y1 = max(cy_prev - pad, 0)
    x2 = min(cx_prev + pad, w_frame - 1)
    y2 = min(cy_prev + pad, h_frame - 1)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # Processamento
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold automático
    #_, thresh = cv2.threshold(
    #gray, 100, 255, cv2.THRESH_BINARY_INV
    #)
    #_, thresh = cv2.threshold(
    #    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    #)

    _, thresh = cv2.threshold(
    gray, 80, 255, cv2.THRESH_BINARY_INV
    )

    # Limpeza de ruído
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Contornos
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 20:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx_prev = int(M["m10"] / M["m00"]) + x1
                cy_prev = int(M["m01"] / M["m00"]) + y1

    # Salva trajetória
    trajetoria.append((cx_prev, cy_prev))

    # Salva dados
    t = frame_idx / fps
    dados.append([frame_idx, t, cx_prev, cy_prev])

    # ===== DESENHOS =====
    # ponto verde (massa)
    cv2.circle(frame, (cx_prev, cy_prev), 6, (0, 255, 0), -1)

    # região de busca (azul)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # trajetória (amarelo)
    for i in range(1, len(trajetoria)):
        cv2.line(frame, trajetoria[i - 1], trajetoria[i], (0, 255, 255), 2)

    # mostrar frame
    cv2.imshow("Tracking Pendulo", frame)

    # ESC para sair
    if cv2.waitKey(10) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# Salvar CSV
df = pd.DataFrame(dados)
df.columns = ["frame", "tempo_s", "x_px", "y_px"]
df.to_csv(output_csv, index=False)

print("\nArquivo salvo:", output_csv)
