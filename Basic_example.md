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
