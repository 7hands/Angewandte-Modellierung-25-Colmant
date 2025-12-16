import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, callbacks
from tensorflow.keras.layers import *
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
        # 1. Sehr schmale Eingangskonvolution (nur 8 Filter)
        Conv2D(4, 3, padding='same', use_bias=False, input_shape=(28,28,1)),
        ReLU(),

        # 2. Depthwise 5×5 (großes Rezeptivfeld mit extrem wenig Parametern)
        DepthwiseConv2D(5, padding='same', use_bias=False),
        ReLU(),

        # 3. 1×1 Bottleneck auf 16 Kanäle
        Conv2D(10, 1, use_bias=False),
        ReLU(),

        # 4. Nochmal Depthwise 5×5
        DepthwiseConv2D(5, padding='same', use_bias=False),
        ReLU(),

        # 5. 1×1 auf 10 Kanäle (direkt die Klassen!)
        Conv2D(10, 1, use_bias=False),

        # 6. Global Average Pooling + Softmax
        GlobalAveragePooling2D(),
        Softmax()
    ], name="456-params-MNIST")

    #model.summary()

    # 3. Training Setup
    # Wir nutzen Learning Rate Decay. Wir starten hoch (0.01) und gehen schnell runter.
    # # Das ist essentiell für kurze Trainings (10 Epochen).
    # lr_schedule = optimizers.schedules.PolynomialDecay(
    #     initial_learning_rate=0.05,
    #     decay_steps=0.005, # Über 10 Epochen
    #     end_learning_rate=0.015,
    #     power=1.2
    # )
    reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',   # Wir beobachten den Validation Loss
    factor=0.6,           # Neue LR = Alte LR * 0.2 (Reduktion auf 20%)
    patience=1,           # Wie viele Epochen warten wir ohne Besserung? (bei nur 10 Epochen niedrig halten!)
    min_lr=0.00008,        # Untergrenze, damit das Netz nicht "einschläft"
    min_delta = 0.012,
    verbose=0             # Gibt eine Meldung aus, wenn die LR reduziert wird
)
    initial_lr = 0.05
    batch_size = 30
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=5*len(train_data)//batch_size,
        t_mul=2.0, m_mul=0.8, alpha=0.001
    )
    model.compile(optimizer=optimizers.Nadam(learning_rate=lr_schedule), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])


    
    parms = model.count_params()
    print(parms)
    # 4. Training
    print("Starte 10-Epochen Sprint...")
          
    history = model.fit(
        x_train, y_train, 
        epochs=10, 
        batch_size=batch_size, 
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr],
        verbose=1
    )
          
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Acc = {acc:.4f}")
    model.summary()
    if acc>=0.98:
        print("big Win!")
        filename = f"acc_{acc:.4f}_params_{parms}.keras"
        model.save(filename)
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
            break
        


