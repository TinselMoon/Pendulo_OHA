import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("pendulo_posicoes.csv")

t = df["tempo_s"].values
x = df["x_px"].values

# remove offset (centraliza)
x = x - np.mean(x)

def modelo(t, A, b, omega, phi):
    return A * np.exp(-b * t) * np.cos(omega * t + phi)

# chutes iniciais
A0 = np.max(x)
b0 = 0.02
omega0 = 4.2   # baseado no cálculo teórico
phi0 = 0

p0 = [A0, b0, omega0, phi0]

params, _ = curve_fit(modelo, t, x, p0=p0, maxfev=20000)

A, b, omega, phi = params

print(f"A = {A:.3f} px")
print(f"b = {b:.5f} 1/s")
print(f"ω = {omega:.5f} rad/s")
print(f"φ = {phi:.3f} rad")

# fator de qualidade
Q = omega / (2 * b)
print(f"Q = {Q:.2f}")

# gráfico
t_fit = np.linspace(t.min(), t.max(), 2000)
x_fit = modelo(t_fit, *params)

plt.plot(t, x, ".", label="dados")
plt.plot(t_fit, x_fit, label="ajuste")
plt.legend()
plt.xlabel("tempo (s)")
plt.ylabel("x (px)")
plt.show()
