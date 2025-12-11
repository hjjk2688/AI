# 인공 신경망 예제

## 7 segment

<img width="481" height="344" alt="image" src="https://github.com/user-attachments/assets/483192c8-dfbc-45be-b9a7-33f73b0f9083" />

<img width="488" height="398" alt="image" src="https://github.com/user-attachments/assets/9fb364d5-9665-46a8-a2ad-607d4f24305b" />

- 입력층: 7개
- 은닉층: 8개
- 출력층: 4개
>  준비된 데이터 중 학습80%, 검증 20%로 사용하지만 간단한 예제라서 데이터를 나누지않았다.

```python
import tensorflow as tf
import numpy as np

np.set_printoptions(precision=4, suppress=True)

X=np.array([
    [ 1, 1, 1, 1, 1, 1, 0 ],  # 0
    [ 0, 1, 1, 0, 0, 0, 0 ],  # 1
    [ 1, 1, 0, 1, 1, 0, 1 ],  # 2
    [ 1, 1, 1, 1, 0, 0, 1 ],  # 3
    [ 0, 1, 1, 0, 0, 1, 1 ],  # 4
    [ 1, 0, 1, 1, 0, 1, 1 ],  # 5
    [ 0, 0, 1, 1, 1, 1, 1 ],  # 6
    [ 1, 1, 1, 0, 0, 0, 0 ],  # 7 
    [ 1, 1, 1, 1, 1, 1, 1 ],  # 8
    [ 1, 1, 1, 0, 0, 1, 1 ]   # 9
    ])
YT=np.array([
    [ 0, 0, 0, 0 ],  
    [ 0, 0, 0, 1 ], 
    [ 0, 0, 1, 0 ], 
    [ 0, 0, 1, 1 ], 
    [ 0, 1, 0, 0 ], 
    [ 0, 1, 0, 1 ], 
    [ 0, 1, 1, 0 ], 
    [ 0, 1, 1, 1 ], 
    [ 1, 0, 0, 0 ], 
    [ 1, 0, 0, 1 ] 
    ])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(7,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')
    
    ])

model.compile(optimizer='adam', loss='mse')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',  min_delta=1e-7,patience=5)

model.fit(X,YT,
          epochs=10000,
          verbose=2,          
          callbacks=[early_stopping])
Y = model.predict(X)
print(Y)

```

<img width="542" height="319" alt="image" src="https://github.com/user-attachments/assets/4e260fc0-2787-47f3-b771-e4ce773e6088" />

### 국소해 문제
<table>
  <tr>
    <td><img width="600" height="315" alt="image" src="https://github.com/user-attachments/assets/528eff0e-7784-4157-968d-60849328d1a6" /></td>
    <td><img width="500" height="377" alt="image" src="https://github.com/user-attachments/assets/d51b2f6b-77d0-46a3-87cd-73cfb0144564" /></td>
  </tr>
</table>

```
앞의 예제에서 인공 신경망 학습이 제대로 되지 않는 경우가 있는데 이런 현상은 국소해의 문제로 발생 합니다. 
예를 들어, 다음 그림에서 신경망의 학습 과정에서 최소값 지점을 찾지 못하고 극소값 지점에 수렴하는 경우입니다.
국소해의 문제가 발생할 경우엔 재학습을 수행해 보거나 은닉층의 노드수를 변경해 봅니다.
여기서는 은닉층 노드의 개수를 16으로 늘려봅니다.
```

### linear (일자 = 1차함수)
```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(7,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='linear')
    ])
```
<img width="558" height="318" alt="image" src="https://github.com/user-attachments/assets/0110cd5e-5e4e-44ac-82ba-a2a62d623f44" />


### 입력, 출력 변경
#### 1. 출력 10진수 한자리 수 변경 (목표값 변경)

<img width="491" height="357" alt="image" src="https://github.com/user-attachments/assets/9b2d043b-73ae-4884-9e31-0fe21a731569" />

```python
YT_1 = np.array([
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9    
    ])

YT = YT_1

model = tf.keras.Sequential([
    tf.keras.Input(shape=(7,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
    ])
--  원래 코드랑 같음 --
```

<img width="559" height="255" alt="image" src="https://github.com/user-attachments/assets/0d11db22-715e-4b1b-bc7a-79c21d8a2c55" />

#### 2. 입력 4개 출력 7개

<img width="933" height="383" alt="image" src="https://github.com/user-attachments/assets/78297db7-274d-459b-8f7f-74f183fe1b6f" />

-  입력된 숫자(10진수) 맞게 7segment 켜기 => '숫자 5에 맞게 7 segment 켜줘'

```python
X, YT = YT, X

model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(7, activation='linear')
    ])
```

<img width="741" height="498" alt="image" src="https://github.com/user-attachments/assets/9230fa5c-d0ab-4f4c-8db6-163e16b3a4a4" />



#### 은닉층 추가

<img width="968" height="397" alt="image" src="https://github.com/user-attachments/assets/5da7d5b7-03e3-4792-b4ac-22683fa8c895" />

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(7,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu')
    tf.keras.layers.Dense(4, activation='linear')    
    ])
```

<img width="571" height="281" alt="image" src="https://github.com/user-attachments/assets/f8d0b775-2914-4a65-864d-08809870ffbb" />

#### 모델 내보내기
```
model.save('model7seg.h5')
```
<img width="707" height="53" alt="image" src="https://github.com/user-attachments/assets/33861f47-7e6e-4c19-b6ea-6bdc1b8ff75d" />


#### 모델 가져오기(추론)
- 학습된 모델을 가져와 추론한다.
```python
import tensorflow as tf
from _7seg_data import X

model=tf.keras.models.load_model('model7seg.keras')
Y = model.predict(X)
print(Y)
```
<img width="525" height="416" alt="image" src="https://github.com/user-attachments/assets/7d838ca7-e301-4f7c-9c71-c344d274576e" />

- 입력을 X[:1]로 변경합니다. 이렇게 하면 X의 0번 항목으로만 구성된 numpy 2차 배열 출력

```python
import tensorflow as tf
from _7seg_data import X

model=tf.keras.models.load_model('model7seg.keras')
Y = model.predict(X[:1])
print(X[:1].shape)
print(Y)
```

<img width="531" height="98" alt="image" src="https://github.com/user-attachments/assets/4e07968c-19d7-418e-96a7-f0f92df899ac" />

---

## MNIST (손글씨 판단)

#### 1. 입력층이 784 인 이유

- MNIST 이미지 한 장의 크기가 가로 28픽셀, 세로 28픽셀
- 이 두 숫자를 곱하면 `28 * 28 = 784` 가 됩니다.

왜 곱해야 하는가? (Flatten 과정)
```
MNIST 이미지는 원래 2차원(28x28) 데이터입니다. 하지만 가장 기본적인 신경망 모델(MLP, 다층 퍼셉트론)에
이 데이터를 입력으로 넣기 위해서는, 2차원 이미지를 1차원의 긴 벡터(한 줄) 형태로 쭉 펼쳐줘야 합니다.
이 과정을 Flatten(평탄화)이라고 부릅니다.
```

## 결과
```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X,YT), (x, yt) = mnist.load_data() #60000개의 학습데이터 , 10000개의 겁증데이터

X, x = X/255, x/255 # 60000x28x28 , 10000x28x28  pixel 값이라서 255 나눠서 0~1사이로 바꾼다 relu사용할려고 줄임 
X, x = X.reshape((60000,784)), x.reshape((10000,784)) # 2차원 데이터를 1차원 데이터로 변경 

model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X,YT,epochs=5)

model.evaluate(x,yt)

```

<img width="1016" height="380" alt="image" src="https://github.com/user-attachments/assets/cb27e4b1-7f8d-4d66-88ee-44f90321fd1e" />

```python
(X, YT), (x, yt) = mnist.load_data()
```

<img width="372" height="207" alt="image" src="https://github.com/user-attachments/assets/859b0607-19b0-44c1-a042-227e883b3454" />

mnist.load_data 함수를 호출하여 손글씨 숫자 데이터를 읽어와 X, YT, x, yt 변수가 가리키게 합니다. 
X, x 변수는 각각 6만개의 학습용 손글씨 숫자 데이터와 1만개의 시험용 손글씨 숫자 데이터를 가리킵니다. 
YT, yt 변수는 각각 6만개의 학습용 손글씨 숫자 라벨과 1만개의 시험용 손글씨 숫자 라벨을 가리킵니다. 
예를 들어 X[0], YT[0] 항목은 각각 다음과 같은 손글씨 숫자 5에 대한 그림과 라벨 5를 가리킵니다. 
또, x[0], yt[0] 항목은 각각 다음과 같은 손글씨 숫자 7에 대한 그림과 라벨 7을 가리킵니다.

```python
X, x = X/255, x/255 # 60000x28x28, 10000x28x28 # 1
X, x = X.reshape((60000,784)), x.reshape((10000,784)) # 2 
```
<table>
    <tr>
        <td><img width="443" height="283" alt="image" src="https://github.com/user-attachments/assets/faee1472-f5ad-41fd-be8d-f3cc6935d7a3" /></td>
        <td><img width="683" height="353" alt="image" src="https://github.com/user-attachments/assets/debf2f86-57bb-4b23-8d8d-0cdd06cc704e" /></td>
    </tr>    
</table>

1. X, x 변수가 가리키는 6만개, 1만개의 그림은 각각 28x28 픽셀로 구성된 그림이며, 1픽셀의 크기는 8비트로 0에서 255사이의 숫자를 가집니다. 
모든 픽셀의 숫자를 255.0으로 나누어 각 픽셀을 0.0에서 1.0사이의 실수로 바꾸어 인공 신경망에 입력하게 됩니다.

2. X, x 변수가 가리키는 6만개, 1만개의 그림은 각각 28x28 픽셀, 28x28 픽셀로 구성되어 있습니다. 
이 예제에서 소개하는 인공 신경망의 경우 그림 데이터를 입력할 때 28x28 픽셀을 784(=28x28) 픽셀로 일렬로 세워서 입력하게 됩니다.

```python
model.valuate(x,yt) 
```
- 학습된 모델 테스트 하기


---
#### data shape 확인하기
```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X,YT), (x, yt) = mnist.load_data()
print(X.shape, YT.shape, x.shape, yt.shape)
```
<img width="521" height="78" alt="image" src="https://github.com/user-attachments/assets/7c44d38f-7839-4f69-862e-addc81a28b78" />

#### 학습 데이터 그림 확인
```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X,YT), (x, yt) = mnist.load_data()
import matplotlib.pyplot as plt

plt.imshow(X[0])
plt.show()

print(YT[0])
print(tf.one_hot(YT[0],10))
```
- 0번 학습 데이터 =  5
  
<img width="696" height="540" alt="image" src="https://github.com/user-attachments/assets/5af99083-1b93-4cd9-9138-d089edab4cd8" />

- matplotlib.imshow()는 색상맵(cmap)을 지정하지 않으면 기본값이 'viridis'
- viridis는 보라 → 초록 → 노랑으로 이어지는 색상 스케일
- 그래서 하나만 출력할 때 cmap을 지정하지 않으면 노란색이 높은 값(글씨 부분), 보라색이 낮은 값(배경)으로 보이는 것.
```python
plt.imshow(data, cmap=plt.cm.binary) # 흰색 검정색 으로 표시
```

-one_hot Encoding: 정수(Integer)로 표현된 범주형 데이터를 0과 1로만 이루어진 벡터로 변환하는 방법

#### 학습 데이터 그림 픽셀 값 출력

```
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X,YT), (x, yt) = mnist.load_data()

for row in range(28):
    for col in range(28):
        print('%4s' %X[0][row][col], end='')
    print()
```
<img width="1181" height="530" alt="image" src="https://github.com/user-attachments/assets/be9dc6e8-3f4f-4c7c-87ea-206dd404eae9" />

#### 학습 데이터 그림 확인2

```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X,YT), (x, yt) = mnist.load_data()
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X[i],cmap=plt.cm.binary)
    plt.xlabel(YT[i])
plt.show()
```

<img width="904" height="836" alt="image" src="https://github.com/user-attachments/assets/39ce867d-4649-47dc-80fa-98c6facc7ed2" />

- plt.subplot(5,5,i+1)  5x5 행렬 - 25개 이미지 생성

#### 학습 모델 저장 및 추론모델로 사용하기
```python
model.save('mnist_model.keras')
```
<img width="691" height="42" alt="image" src="https://github.com/user-attachments/assets/ef6901cf-220a-461a-994d-c5405b5dce05" />

