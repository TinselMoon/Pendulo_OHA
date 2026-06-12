import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit #encontra a função da curva

df = pd.read_csv("pendulo_posicoes.csv")

t = df["tempo_s"].values
x = df["x_px"].values

x = x - np.mean(x) #centraliza os valores de oscilação, como não há uma referência no vídeo, ajuda muito

def modelo(t, A, b, omega, phi):
    return A * np.exp(-b * t) * np.cos(omega * t + phi)

#chutes para o curve_fit encontrar os valores da função
A0 = np.max(x) #procura x max
b0 = 0.02 #chuta valor baixo p coeficiente de amortecimento
omega0 = 5.5   # baseado no cálculo teórico
phi0 = 0

p0 = [A0, b0, omega0, phi0]

params, _ = curve_fit(modelo, t, x, p0=p0, maxfev=20000) #testa os valores até se encaixarem

A, b, omega, phi = params

print(f"A = {A:.3f} px")
print(f"b = {b:.5f} 1/s")
print(f"ω = {omega:.5f} rad/s")
print(f"φ = {phi:.3f} rad")

# fator de qualidade
Q = omega / (2 * b)
print(f"Q = {Q:.2f}")

# gráfico
t_fit = np.linspace(t.min(), t.max(), 3600)
x_fit = modelo(t_fit, *params)

plt.plot(t, x, ".", label="dados")
plt.plot(t_fit, x_fit, label="ajuste")
plt.legend()
plt.xlabel("tempo (s)")
plt.ylabel("x (px)")
plt.show()
