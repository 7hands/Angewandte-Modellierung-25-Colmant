import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np

def trainfunc():
    # 1. Daten laden
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # WICHTIG: Mean/Std Normalisierung hilft dem Netz schneller zu konvergieren als nur /255
    # mean, std = 0.1307, 0.3081
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # 2. Architektur
    model = models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(12, (3, 3), padding='valid', use_bias=True, activation= "relu"),
        layers.MaxPooling2D((2, 2)), 
        
        
        # Layer 3: Bottleneck vor dem Output
        layers.Conv2D(12, (3, 3), padding='valid', use_bias=True, activation= "relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        
        layers.Dense(10, activation='softmax', use_bias=True)
    ])

    #model.summary()

    # 3. Training Setup
    # Wir nutzen Learning Rate Decay. Wir starten hoch (0.01) und gehen schnell runter.
    # Das ist essentiell für kurze Trainings (10 Epochen).
    lr_schedule = optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.1,
        decay_steps=10 * (len(x_train) // 256), # Über 10 Epochen
        end_learning_rate=0.001,
        power=1.2
    )

    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    parms = model.count_params()
    print(parms)
    # 4. Training
    print("Starte 10-Epochen Sprint...")
    history = model.fit(
        x_train, y_train, 
        epochs=10, 
        batch_size=256, 
        validation_data=(x_test, y_test),
        verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Acc = {acc:.4f}")
    model.summary()
    if acc>=0.98:
    #     print("big Win!")
    #     filename = f"256normal_acc_{acc:.4f}_params_{parms}.keras"
    #     model.save(filename)
        return acc



if __name__ == '__main__':
    results = []
    for i in range(100):
        res = None
        print(f"run: {i}")
        res = trainfunc()
        if res != None:
            print("we got 'im")
            results.append((res, i))
        print(results)

