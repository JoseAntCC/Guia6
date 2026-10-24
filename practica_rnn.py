"""
Laboratorio 06 - RNN simple vs LSTM en PyTorch (series temporales)

Ejecuta todo en orden:
  - Actividad 1: inspeccion de dimensiones (tensores 3D)
  - Actividad 2: hidden_size 16 vs 64 y numero de parametros
  - Actividad 3: SEQ_LEN 10 vs 50 con RNN simple
  - Reto: RNN vs LSTM con SEQ_LEN = 50 (curvas de perdida y predicciones)

Genera las imagenes: loss_seq10_vs_seq50.png, comparacion_loss_rnn_lstm.png,
predicciones_rnn_lstm.png y el archivo resultados.txt con los numeros finales.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")  # guarda las figuras sin necesitar ventana
import matplotlib.pyplot as plt

SEED = 42
EPOCHS = 100
LR = 0.01
RESULTADOS = []  # lineas que se guardan en resultados.txt


def log(texto=""):
    print(texto)
    RESULTADOS.append(texto)


def fijar_semilla(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. Datos sinteticos (onda senoidal)
# ---------------------------------------------------------------------------
def generar_datos(n_puntos=600, seq_len=10):
    t = np.linspace(0, 50, n_puntos)
    serie = np.sin(t)

    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i + seq_len])
        y.append(serie[i + seq_len])

    # [N, seq_len, 1] y [N, 1]
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X, y


def dividir(X, y, prop_train=0.8):
    corte = int(len(X) * prop_train)
    return X[:corte], y[:corte], X[corte:], y[corte:]


# ---------------------------------------------------------------------------
# 2. Modelos
# ---------------------------------------------------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)          # [batch, seq_len, hidden]
        out_final = out[:, -1, :]     # Many-to-One: ultimo paso temporal
        return self.fc(out_final)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)  # estado oculto y estado de celda
        out_final = out[:, -1, :]
        return self.fc(out_final)


def contar_parametros(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 3. Entrenamiento / evaluacion
# ---------------------------------------------------------------------------
def entrenar(model, X, y, epochs=EPOCHS, lr=LR, nombre="modelo"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    historial = []
    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        historial.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  [{nombre}] Epoch [{epoch + 1}/{epochs}], "
                  f"Loss: {loss.item():.6f}")
    return historial


def evaluar(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        mse = nn.MSELoss()(pred, y).item()
    return mse, pred


def experimento(clase_modelo, seq_len, hidden_size=16, nombre="modelo"):
    """Entrena un modelo y devuelve (historial, mse_train, mse_test, pred, y_test)."""
    fijar_semilla()
    X, y = generar_datos(n_puntos=600, seq_len=seq_len)
    X_tr, y_tr, X_te, y_te = dividir(X, y)
    model = clase_modelo(input_size=1, hidden_size=hidden_size, output_size=1)
    hist = entrenar(model, X_tr, y_tr, nombre=nombre)
    mse_tr, _ = evaluar(model, X_tr, y_tr)
    mse_te, pred_te = evaluar(model, X_te, y_te)
    return hist, mse_tr, mse_te, pred_te, y_te


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ----- Actividad 1: dimensiones -----
    log("=" * 60)
    log("ACTIVIDAD 1: Inspeccion de dimensiones")
    log("=" * 60)
    fijar_semilla()
    X, y = generar_datos(n_puntos=600, seq_len=10)
    log(f"X.shape = {tuple(X.shape)}  -> [Batch Size, Sequence Length, Features]")
    log(f"y.shape = {tuple(y.shape)}  -> [Batch Size, 1]")
    log("  Eje 0 (Batch Size): numero de ventanas/muestras (600 - 10 = 590).")
    log("  Eje 1 (Sequence Length): pasos temporales por ventana (10).")
    log("  Eje 2 (Features): variables por paso (1: el valor de la senoide).")

    # ----- Actividad 2: hidden_size 16 vs 64 -----
    log("")
    log("=" * 60)
    log("ACTIVIDAD 2: Parametros entrenables segun hidden_size")
    log("=" * 60)
    for h in (16, 64):
        n_rnn = contar_parametros(SimpleRNNModel(1, h, 1))
        n_lstm = contar_parametros(LSTMModel(1, h, 1))
        log(f"hidden_size={h:>3} | RNN: {n_rnn:>6} parametros | "
            f"LSTM: {n_lstm:>6} parametros")
    log("  RNN  : h*(in+h) + 2h  (+ capa fc: h+1)")
    log("  LSTM : 4*[h*(in+h) + 2h]  (+ capa fc: h+1)")

    # ----- Actividad 3: SEQ_LEN 10 vs 50 (RNN simple) -----
    log("")
    log("=" * 60)
    log("ACTIVIDAD 3: Efecto de SEQ_LEN en la RNN simple")
    log("=" * 60)
    hist_10, tr10, te10, _, _ = experimento(SimpleRNNModel, 10, nombre="RNN seq=10")
    hist_50, tr50, te50, _, _ = experimento(SimpleRNNModel, 50, nombre="RNN seq=50")
    log(f"RNN SEQ_LEN=10 | loss final train={hist_10[-1]:.6f} | MSE test={te10:.6f}")
    log(f"RNN SEQ_LEN=50 | loss final train={hist_50[-1]:.6f} | MSE test={te50:.6f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(hist_10, label="RNN SEQ_LEN=10")
    plt.plot(hist_50, label="RNN SEQ_LEN=50")
    plt.yscale("log")
    plt.xlabel("Epoca")
    plt.ylabel("Loss (MSE, escala log)")
    plt.title("RNN simple: efecto de la longitud de secuencia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_seq10_vs_seq50.png", dpi=150)
    plt.close()

    # ----- Reto: RNN vs LSTM con SEQ_LEN = 50 -----
    log("")
    log("=" * 60)
    log("RETO: RNN simple vs LSTM con SEQ_LEN = 50 (mismas condiciones)")
    log("=" * 60)
    hist_rnn, tr_r, te_r, pred_r, y_te = experimento(
        SimpleRNNModel, 50, nombre="RNN")
    hist_lstm, tr_l, te_l, pred_l, _ = experimento(
        LSTMModel, 50, nombre="LSTM")
    log(f"RNN  | loss final train={hist_rnn[-1]:.6f} | MSE test={te_r:.6f}")
    log(f"LSTM | loss final train={hist_lstm[-1]:.6f} | MSE test={te_l:.6f}")
    mejor = "LSTM" if te_l < te_r else "RNN simple"
    log(f"Mejor ajuste en test: {mejor}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(hist_rnn, label="RNN simple")
    plt.plot(hist_lstm, label="LSTM")
    plt.yscale("log")
    plt.xlabel("Epoca")
    plt.ylabel("Loss (MSE, escala log)")
    plt.title("Convergencia con SEQ_LEN=50")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparacion_loss_rnn_lstm.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(y_te.squeeze().numpy(), label="Real", color="black", lw=2)
    plt.plot(pred_r.squeeze().numpy(), label="RNN simple", linestyle="--")
    plt.plot(pred_l.squeeze().numpy(), label="LSTM", linestyle=":")
    plt.xlabel("Muestra de test")
    plt.ylabel("sin(t)")
    plt.title("Predicciones en el conjunto de prueba (SEQ_LEN=50)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("predicciones_rnn_lstm.png", dpi=150)
    plt.close()

    with open("resultados.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(RESULTADOS))
    print("\nListo. Se generaron las graficas .png y resultados.txt")
