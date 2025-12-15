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

---

### Alexnet 신경망 랜덤 데이터 테스트

```python
import tensorflow as tf
import numpy as np

X =np.random.randint(0,256,(5000,227,227,3))
YT=np.random.randint(0,1000,(5000,))
x =np.random.randint(0,256,(1000,227,227,3))
yt=np.random.randint(0,1000,(1000,))


model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(227,227,3)),
    tf.keras.layers.Conv2D(96,(11,11),(4,4),activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Conv2D(256,(5,5),(1,1),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Conv2D(384,(3,3),(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(384,(3,3),(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(256,(3,3),(1,1),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096,activation='relu'),
    tf.keras.layers.Dense(4096,activation='relu'),
    tf.keras.layers.Dense(1000,activation='softmax')
])

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(X,YT,epochs=5)
model.evaluate(x,yt)
```

<img width="960" height="256" alt="image" src="https://github.com/user-attachments/assets/8ecfa335-c557-4f6b-8bbd-c7aef8717fd1" />

---

## Alexnet 신경망 cifar10 테스
```python
import tensorflow as tf

mnist=tf.keras.datasets.cifar10
(X,YT),(x,yt)=mnist.load_data()
X,x=X/255,x/255

model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),
    tf.keras.layers.Conv2D(96,(5,5),(1,1),activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Conv2D(56,(5,5),(1,1),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Conv2D(84,(3,3),(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(84,(3,3),(1,1),padding='same',activation='relu'),
    tf.keras.layers.Conv2D(56,(3,3),(1,1),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3),(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(496,activation='relu'),
    tf.keras.layers.Dense(496,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax'),
])

model.summary()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(X,YT,epochs=5)
model.evaluate(x,yt)

```

<table>
    <tr>
        <td><img width="827" height="559" alt="image" src="https://github.com/user-attachments/assets/8d5d2f4c-e06c-4bf4-bb70-f6e0bb13b86d" /></td>
        <td><img width="937" height="215" alt="image" src="https://github.com/user-attachments/assets/7fb3ef19-ae32-403e-a670-8002d6265e44" /></td>
    </tr>
    
</table>

---

## 배치 정규화(BatchNormalization)
배치 정규화(batch normalization)는 딥 러닝 모델의 학습을 안정화하고 속도를 높이는 기술 중 하나입니다.

```
각 층의 활성화 함수 입력값을 정규화
(normalization)하는 방법입니다. 각 층의 입력값을 평균과 분산으로 정규화한 후, 학습
가능한 두 개의 파라미터인 스케일(scale)과 시프트(shift)를 이용해 새로운 활성화 함수
입력값을 계산합니다. 이렇게 하면, 각 층의 입력 분포가 비슷해지므로 학습이 안정적으로
이루어지고, 모델의 일반화 성능이 향상됩니다.
```

<img width="415" height="332" alt="image" src="https://github.com/user-attachments/assets/c1df407c-f3bd-456b-babf-efd59559907f" />

<img width="879" height="445" alt="image" src="https://github.com/user-attachments/assets/5f735f37-878c-40f1-af30-321dad8f9109" />


```python

model=tf.keras.Sequential([
 tf.keras.layers.Input(shape=(32,32,3)),
 tf.keras.layers.Conv2D(96,(5,5),(1,1)),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.ReLU(),
 tf.keras.layers.MaxPooling2D((3,3),(2,2)),
 tf.keras.layers.Conv2D(56,(5,5),(1,1),padding='same'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.ReLU(),
 tf.keras.layers.MaxPooling2D((3,3),(2,2)),
 tf.keras.layers.Conv2D(84,(3,3),(1,1),padding='same'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.ReLU(),
 tf.keras.layers.Conv2D(84,(3,3),(1,1),padding='same'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.ReLU(),
 tf.keras.layers.Conv2D(56,(3,3),(1,1),padding='same'),
 tf.keras.layers.BatchNormalization(),
 tf.keras.layers.ReLU(),
 tf.keras.layers.MaxPooling2D((3,3),(2,2)),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(496,activation='relu'),
 tf.keras.layers.Dense(496,activation='relu'),
 tf.keras.layers.Dense(10,activation='softmax')
])

```

