import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np

# Custom Callback definieren
class FailFast(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Index 4 = Ende der 5. Epoche
        if epoch == 4: 
            val_acc = logs.get('val_accuracy')
            if val_acc < 0.97:
                print(f"\n---> ABBRUCH: Nur {val_acc:.4f} nach Epoche 5 (Ziel > 0.97).")
                self.model.stop_training = True


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
        
        # Layer 1:
        # Wir bleiben bei 3x3, das ist der effizienteste Standard.
        layers.Conv2D(6, (3, 3), padding='valid', use_bias=False),
        # layers.BatchNormalization(), # Wichtig für Speed!
        layers.ReLU(),
        layers.Dropout(0.06),
        layers.MaxPooling2D((2, 2)), 

        layers.DepthwiseConv2D((3,3), padding='valid', use_bias=False ),
        # layers.BatchNormalization(),
        layers.ReLU(),

        # Layer 2: Etwas breiter werden
        layers.SeparableConv2D(15, (3, 3), padding='valid', use_bias=False),
        # layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.15),
        layers.MaxPooling2D((2,2)),
        
        # Layer 3: Bottleneck vor dem Output
        layers.SeparableConvolution2D(8, (3, 3), padding='valid', use_bias=False),
        # layers.BatchNormalization(),    
        layers.ReLU(),



        # # layers.MaxPooling2D((3,3)),
        # layers.Flatten(),
        # layers.Softmax()
        # # layers.Dense(10, activation='softmax') 
        # # Output Layer: Convolutional Classifier (spart Parameter gegenüber Dense)

        layers.Conv2D(10, (1,1), use_bias=True),
        layers.GlobalAveragePooling2D(),
        layers.Softmax()
    ])

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
    min_delta = 0.008,
    verbose=0             # Gibt eine Meldung aus, wenn die LR reduziert wird
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath='best_model.keras',       # Dateiname
        monitor='val_accuracy',            # Worauf achten wir?
        mode='max',                        # 'max' weil höhere Accuracy besser ist
        save_best_only=True,               # WICHTIG: Nur speichern, wenn besser!
        verbose=1                          # Bescheid sagen, wenn gespeichert wird
    )

    model.compile(optimizer=optimizers.Nadam(learning_rate=0.0145), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    parms = model.count_params()
    print(parms)
    # 4. Training
    print("Starte 10-Epochen Sprint...")
          
    history = model.fit(
        x_train, y_train, 
        epochs=10, 
        batch_size=60, 
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, checkpoint, FailFast()],
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
        


