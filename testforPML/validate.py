import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# 1. MNIST Test-Daten laden
# Wir brauchen hier nur den Test-Teil (x_test, y_test)
(_, _), (x_test, y_test) = mnist.load_data()

# 2. Daten vorverarbeiten (Preprocessing)
# WICHTIG: Dies muss exakt so geschehen, wie du es beim Training gemacht hast.

# A) Normalisierung: Pixelwerte von 0-255 auf 0-1 skalieren
x_test = x_test.astype('float32') / 255.0

# B) Reshaping: CNNs erwarten 4 Dimensionen (Batch, Höhe, Breite, Kanäle)
# MNIST ist schwarz-weiß, also 1 Kanal.
x_test = x_test.reshape((-1, 28, 28, 1))

# Optional: Falls dein Modell One-Hot-Encoding erwartet (categorical_crossentropy),
# musst du y_test umwandeln. Meistens wird aber sparse_categorical_crossentropy genutzt,
# dann brauchst du diesen Schritt NICHT.
# y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. Das .keras Modell laden
try:
    model = tf.keras.models.load_model('256normal_acc_0.9803_params_822.keras')
    print("Modell erfolgreich geladen.")
except OSError:
    print("Fehler: Die Datei wurde nicht gefunden. Überprüfe den Pfad.")

# 4. Evaluieren
print("\nStarte Evaluierung...")
results = model.evaluate(x_test, y_test, verbose=1)

# 5. Ergebnisse ausgeben
print("------------------------------------------------")
print(f"Test Loss:     {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")
print("------------------------------------------------")