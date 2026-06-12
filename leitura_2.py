import cv2 #visao
import numpy as np #matematica
import pandas as pd #csv

video_path = "pendulo_preto.mp4"
output_csv = "pendulo_posicoes.csv"

x0, y0 = 1100, 720
w, h = 100, 100
pad = 80  # area ao redor do ponto

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Erro ao abrir vídeo")

fps = cap.get(cv2.CAP_PROP_FPS) #num fps do video
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #total de frames

print(f"FPS: {fps}")
print(f"Total de frames: {frame_count}")

dados = [] #tempo, pos x, pos y

cx_prev = x0 + w // 2
cy_prev = y0 + h // 2

trajetoria = [] #pontos lidos a cada frame

frame_idx = 0 #contado de frames

while True:
    ret, frame = cap.read()
    if not ret:
        break #termina o loop quando o video acaba

    h_frame, w_frame = frame.shape[:2] #altura e largura do video

    # Região de busca
    x1 = max(cx_prev - pad, 0)
    y1 = max(cy_prev - pad, 0)
    x2 = min(cx_prev + pad, w_frame - 1) #limita a busca nas bordas da imagem
    y2 = min(cy_prev + pad, h_frame - 1) #limita a busca nas bordas da imagem

    crop = frame[y1:y2, x1:x2] #recorta a imagem na parte da busca

    if crop.size == 0:
        continue

    # Processamento
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) #aqui converte a imagem para a escala do cinza
    gray = cv2.GaussianBlur(gray, (5, 5), 0) #aplica o desfoque gaussiano, diz-se que ajuda na detecção

    # Threshold automático
    #_, thresh = cv2.threshold(
    #gray, 100, 255, cv2.THRESH_BINARY_INV
    #)
    #_, thresh = cv2.threshold(
    #    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    #)

    _, thresh = cv2.threshold(
    gray, 80, 255, cv2.THRESH_BINARY_INV
    ) #Transforma a parte branca do video em escura, e o pendulo preto em branco

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #tratamentos da imagem

    # encontra os contornos dos objetos brancos
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    #calcular as coordenadas do centro do pendulo
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 20:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx_prev = int(M["m10"] / M["m00"]) + x1
                cy_prev = int(M["m01"] / M["m00"]) + y1

    # adiciona os valores na trajetória
    trajetoria.append((cx_prev, cy_prev))

    # calcula o tempo 
    t = frame_idx / fps
    # salva toso os valores na lista de dados para enviar ao arquivo .csv
    dados.append([frame_idx, t, cx_prev, cy_prev])

    # desenha o ponto verde na massa
    cv2.circle(frame, (cx_prev, cy_prev), 6, (0, 255, 0), -1)

    # faz o quadrado azul na área de busca
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # desenha a trajetória amarela
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

df = pd.DataFrame(dados) #transforma a lista em uma tabela p salvar no .csv
df.columns = ["frame", "tempo_s", "x_px", "y_px"]
df.to_csv(output_csv, index=False)

print("\nArquivo salvo:", output_csv)
