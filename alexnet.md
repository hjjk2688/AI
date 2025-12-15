# Alexnet

<img width="869" height="393" alt="image" src="https://github.com/user-attachments/assets/5a187e6e-c85b-4e29-96c8-0622761221bf" />

```python
import tensorflow as tf


model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(227,227,3)),
    tf.keras.layers.Conv2D(96,(11,11),(4,4))
    
    ])

model.summary()

```

<img width="573" height="186" alt="image" src="https://github.com/user-attachments/assets/3d915c4c-8b37-437a-834a-c763e27a3fd9" />

```python
model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(227,227,3)),
    tf.keras.layers.Conv2D(96,(11,11),(4,4)),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Conv2D(256,(5,5),(1,1),padding='same'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Conv2D(384,(3,3),(1,1),padding='same'),
    tf.keras.layers.Conv2D(384,(3,3),(1,1),padding='same'),
    tf.keras.layers.Conv2D(256,(3,3),(1,1),padding='same'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096),
    tf.keras.layers.Dense(4096),
    tf.keras.layers.Dense(1000)
])
```

<img width="622" height="461" alt="image" src="https://github.com/user-attachments/assets/cb7b74bf-fcbf-4a96-b240-f7fab5653396" />
