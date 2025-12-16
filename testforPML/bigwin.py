import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import os
import concurrent.futures

# Konfiguration
RUNS_TOTAL = 100        # Wie viele Versuche insgesamt?
PARALLEL_WORKERS = 5   # Wie viele gleichzeitig? (Setze dies auf Anzahl deiner CPU-Kerne - 2)
EPOCHS = 10

def train_single_run(run_id):
    """
    Diese Funktion läuft in einem eigenen Prozess auf einem CPU-Kern.
    """
    # 1. GPU für diesen Prozess DEAKTIVIEREN (Wichtig, sonst Crash!)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # TF Import innerhalb des Prozesses (für Windows Stabilität)
    import tensorflow as tf
    
    # Daten laden (jeder Prozess lädt sie kurz selbst, ist bei MNIST okay)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0) - 0.1307
    x_test = (x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0) - 0.1307
    
    # Architektur bauen
    model = models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        
        # Layer 1: Wenige Filter, aber räumliche Details erhalten
        # Wir bleiben bei 3x3, das ist der effizienteste Standard.
        layers.Conv2D(6, (4, 3), padding='valid', use_bias=False),
        layers.BatchNormalization(), # Wichtig für Speed!
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)), # 28x28 -> 14x14
        
        # Layer 2: Etwas breiter werden
        layers.Conv2D(4, (3, 3), padding='valid', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)), # 14x14 -> 6x6
        
        # Layer 3: Bottleneck vor dem Output
        layers.Conv2D(9, (3, 3), padding='valid', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # Jetzt sind wir bei ca 4x4
        
        # Output Layer: Convolutional Classifier (spart Parameter gegenüber Dense)
        layers.Conv2D(10, (1, 1), use_bias=True),
        layers.GlobalAveragePooling2D(),
        layers.Softmax()
    ])
    print(model.count_params())
    # Kompilieren
    model.compile(optimizer='nadam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Trainieren (Stumm)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=64, verbose=0)
    
    # Evaluieren
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Wenn das Ergebnis gut ist, speichern wir es mit unique ID
    if acc > 0.97:
        filename = f"model_run_{run_id}_acc_{acc:.4f}.keras"
        model.save(filename)
        return run_id, acc, filename
    
    return run_id, acc, None

if __name__ == '__main__':
    # Dieser Block ist Pflicht für Windows Multiprocessing!
    print(f"Starte {RUNS_TOTAL} Runs auf {PARALLEL_WORKERS} CPU-Kernen parallel...")
    
    best_overall_acc = 0.0
    results = []

    # ProcessPoolExecutor verwaltet die Prozesse für uns
    with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # Aufträge verteilen
        futures = [executor.submit(train_single_run, i) for i in range(RUNS_TOTAL)]
        
        # Ergebnisse einsammeln, sobald sie fertig werden
        for future in concurrent.futures.as_completed(futures):
            run_id, acc, filename = future.result()
            
            print(f"Run {run_id:03d} fertig -> Acc: {acc*100:.2f}%", end="")
            
            if acc > best_overall_acc:
                best_overall_acc = acc
                print(f" 🏆 NEUER REKORD (Gesamt)!")
            elif filename:
                print(f" (Gut, gespeichert)")
            else:
                print()

    print("-" * 40)
    print(f"Bester Run aller Zeiten: {best_overall_acc*100:.2f}%")