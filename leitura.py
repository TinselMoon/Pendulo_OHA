import cv2
import numpy as np
import pandas as pd

video_path = "pendulo.mp4"
output_csv = "pendulo_posicoes.csv"


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}")
print(f"Frames: {frame_count}")

#ret, first_frame = cap.read()
#if not ret:
#    raise RuntimeError("Não foi possível ler o primeiro frame.")

x0 = 673
y0 = 665
w = 68
h = 75

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

dados = []

pad = 40

trajetoria = [] #pontos lidos a cada frame

cx_prev = x0 + w // 2
cy_prev = y0 + h // 2

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    x1 = max(cx_prev - pad, 0)
    y1 = max(cy_prev - pad, 0)
    x2 = min(cx_prev + pad, frame.shape[1] - 1)
    y2 = min(cy_prev + pad, frame.shape[0] - 1)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        x_center = cx_prev
        y_center = cy_prev
    else:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < 10:
            x_center = cx_prev
            y_center = cy_prev
        else:
            M = cv2.moments(c)
            if M["m00"] != 0:
                x_center = int(M["m10"] / M["m00"]) + x1
                y_center = int(M["m01"] / M["m00"]) + y1
            else:
                x_center = cx_prev
                y_center = cy_prev

    cx_prev, cy_prev = x_center, y_center

    trajetoria.append((cx_prev, cy_prev))
    t = frame_idx / fps
    dados.append([frame_idx, t, x_center, y_center])

    for i in range(1, len(trajetoria)):
        cv2.line(frame, trajetoria[i - 1], trajetoria[i], (0, 255, 255), 2)

    cv2.imshow("Tracking Pendulo", frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(dados, columns=["frame", "tempo_s", "x_px", "y_px"])
df.to_csv(output_csv, index=False)

print(df.head())
print(f"\nArquivo salvo em: {output_csv}")
