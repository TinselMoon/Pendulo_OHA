import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pendulo_posicoes.csv")

print("Desvio padrão de x:", df["x_px"].std())
print("Desvio padrão de y:", df["y_px"].std())

plt.plot(df["tempo_s"], df["x_px"])
plt.xlabel("tempo (s)")
plt.ylabel("x (px)")
plt.show()

plt.plot(df["tempo_s"], df["y_px"])
plt.xlabel("tempo (s)")
plt.ylabel("y (px)")
plt.show()
