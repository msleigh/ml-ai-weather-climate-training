import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

x = np.linspace(-20, 20, 2000)
y = np.sin(x) / (1.0 + x**2)**0.3

plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel(r"$\sin(x)/(1+x^2)^{0.3}$")
plt.title("Test curve")
plt.tight_layout()
plt.savefig("03_plot_curve.png", dpi=150)
plt.close()

print("Saved 03_plot_curve.png")
