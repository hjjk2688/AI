# TensorFlow
## TensorFlow 와 Keras

케라스(Keras)가 텐서플로우의 공식적인 고급 API(High-level API)

- 텐서플로우 (TensorFlow): 저수준(low-level)의 복잡한 연산(행렬 곱, 미분 등)을 수행하는 강력한
  '엔진'입니다.
- 케라스 (Keras): 이 복잡한 엔진을 사용자가 쉽게 조작할 수 있도록 만든 '사용자 인터페이스'입니다.
  model.add(), model.compile(), model.fit()처럼 간단하고 직관적인 명령어로 모델을 만들고 학습시킬 수 있게 해줍니다.

- w(가중치) - 2차 배열(행렬): 일반적으로 (출력 뉴런 개수,) 형태의 shape을 가집니다
  -  가중치는 이전 계층의 모든 뉴런과 현재 계층의 모든 뉴런을 연결하는 역할을 합니다. => 이전계층 뉴런 10개, 현재계층 뉴런 3개 = 총 30개 연결선 = (10, 3)
- b(편향) - 1차 배열(벡터): 일반적으로 (입력 뉴런 개수, 출력 뉴런 개수) 형태의 shape을 가집니다.
  - 편향은 현재 계층의 각 뉴런에 하나씩 더해지는 값입니다.
- shape: 행렬의 차원 ( , ) ',' 로 표시
---
## Shape: 배열의 차원과 크기

`shape`은 배열(행렬)이 각 차원(dimension)에 몇 개의 원소(element)를 가지고 있는지를 알려주는 튜플(tuple)입니다.

---

#### 1. 1차원 배열 (벡터, Vector)

*   **Shape:** `(3,)`
*   **의미:** 1개의 차원에 3개의 원소를 가집니다.
*   **예시:**
    ```python
    import numpy as np
    arr = np.array([10, 20, 30])
    print(arr.shape)  # 출력: (3,)
    ```
    ```
    [10, 20, 30]
    ```
*   **참고:** `(3)`이 아니라 `(3,)`처럼 쉼표(`,`)가 있는 이유는, 파이썬에서 `(3)`은 그냥 숫자 3이지만 `(3,)`은 원소가 하나인 튜플을 의미하기 때문입니다.



#### 2. 2차원 배열 (행렬, Matrix)

*   **Shape:** `(2, 4)`
*   **의미:** 2개의 **행(row)**과 4개의 **열(column)**을 가집니다. 첫 번째 숫자는 행의 개수, 두 번째 숫자는 열의 개수입니다.
*   **예시:**
    ```python
    import numpy as np
    arr = np.array([
        [1, 2, 3, 4],  # 1행
        [5, 6, 7, 8]   # 2행
    ])
    print(arr.shape)  # 출력: (2, 4)
    ```
    ```
    [[1, 2, 3, 4],
     [5, 6, 7, 8]]
    ```
#### 3. 3차원 배열 (텐서, Tensor)

이 개념을 확장하면 3차원 이상의 텐서도 표현할 수 있습니다.

*   **Shape:** `(2, 3, 4)`
*   **의미:** `3행 4열`짜리 행렬이 `2`개 있는 형태입니다. 보통 (깊이, 행, 열) 또는 (채널, 높이, 너비) 순서로 해석합니다.
*   **예시:**
    ```python
    import numpy as np
    arr = np.array([
        [              # 0번째 행렬
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        [              # 1번째 행렬
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24]
        ]
    ])
    print(arr.shape)  # 출력: (2, 3, 4)
    ```

| 차원 | Shape 예시 | 설명 |
| :--- | :--- | :--- |
| **1차원** (벡터) | `(n,)` | n개의 원소를 가진 배열 |
| **2차원** (행렬) | `(m, n)` | m개의 행과 n개의 열을 가진 행렬 |
| **3차원** (텐서) | `(d, m, n)` | m x n 행렬이 d개 있는 묶음 |

### 추가설명

* `tf.keras.Input(shape=(2,))`의 의미:
   * "이 모델에 들어올 데이터(X)의 샘플 하나하나는 2개의 숫자로 이루어진 1차원 배열(벡터) 형태입니다."
     라는 뜻입니다.
   * 이것은 가중치(w)의 모양이 아니라, 모델에 넣어줄 입력 데이터 `X`의 모양을 말하는 것입니다.
   * 예를 들어, X 데이터는 이런 모습일 것입니다: [ [1.2, 3.4], [5.6, 7.8], [9.0, 1.1], ... ]


---
### Dense

#### tf.keras.layers.Dense(2)의 정확한 의미

`Dense`는 '완전 연결 계층(Fully Connected Layer)'이라는 종류의 '층(Layer)'을 의미합니다. 모델을 구성하는 벽돌 한 장과 같은 부품입니다.
그리고 괄호 안의 숫자 2는 해당 Dense 층에 뉴런(neuron)이 몇 개 들어있는지를 의미합니다.
이 2개의 뉴런이 각각 하나의 값을 출력하므로, 결과적으로 이 층의 출력(`output`)은 2개의 값을 가지게 됩니다.

> 즉, Dense(2)는 "2개의 출력을 내보내는 완전 연결 계층"이라는 의미의 부품(layer)입니다.

#### Dense 층의 역할: 중간층 vs 출력층

이 Dense 층은 모델의 어디에 위치하느냐에 따라 '중간층(Hidden Layer)'이 될 수도 있고, '출력층(Output Layer)'이 될 수도 있습니다.

- 예시 1: Dense(2)가 최종 출력층인 경우


```python
model = keras.Sequential([
    layers.Dense(10, activation='relu'),  # 10개의 출력을 내는 중간층
    layers.Dense(5, activation='relu'),   # 5개의 출력을 내는 중간층
    layers.Dense(2, activation='softmax') # 최종적으로 2개의 출력을 내는 출력층
])
```
위 모델에서는 Dense(2)가 가장 마지막에 쓰였으므로, 이 모델의 최종 출력은 2개의 값을 가지게 됩니다. (예:개/고양이 분류 문제에서 각각의 확률)

- 예시 2: Dense(2)가 중간층인 경우

```python
model = keras.Sequential([
    layers.Dense(10, activation='relu'), # 10개의 출력을 내는 중간층
    layers.Dense(2, activation='relu'),  # 2개의 출력을 내는 중간층
    layers.Dense(1, activation='sigmoid') # 최종적으로 1개의 출력을 내는 출력층
])
```
위 모델에서는 Dense(2)가 중간에 쓰였으므로, 중간 계산 과정에서 2개의 값을 만들어 다음 층으로 넘겨주는 역할을 합니다.

> Dense는 '출력 그 자체'가 아니라, '특정 개수의 출력을 만들어내는 층' 입니다.

---

## 코드 분석
```tensorflow
model.compile(optimizer='sgd', loss='mse')
```
- optimizer: 최적화 함수 - 확률적 경사하강법(sgd)
  - 손실 함수가 만든 지형을 보고, 가장 낮은 곳으로 가기 위해 가중치를 수정하는 '행동 규칙'
  - (예: 한 걸음씩 조심해서 내려가기(SGD), 관성을 이용해 미끄러져 내려가기(Momentum), 지형에 맞춰 보폭을 조절하며 내려가기(Adam))
- loss: 손실함수 - 평균제곱오차(mse)
  - 현재 모델이 얼마나 잘못되었는지(오차)를 측정하는 '평가 기준'입니다. 
 
```tensorflow
 model.fit(X,YT,epochs=999)
```
- 모델학습을 시작 하는 함수(딥러닝 7공식 실행시키는 단계)

#### 모델 학습 과정 
1. model.compile(): 내비게이션 목적지 및 경로 설정
- 어떻게 학습할지에 대한 '규칙'과 '전략'을 설정하는 과정입니다.

```
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
```

* `loss='mse'` (손실 함수): 최종 목적지를 설정합니다.
   * "우리의 목적지는 평균 제곱 오차(mse)가 가장 낮은 지점이다."

* `optimizer='sgd'` (옵티마이저): 목적지까지 갈 방법을 선택합니다.
   * "목적지까지 확률적 경사 하강법(sgd)이라는 방법으로 이동하겠다." (일반 국도로 가기)

* `metrics=['accuracy']` (평가 지표): 가는 길에 추가로 확인할 정보를 설정합니다.
   * "이동하면서 정확도(accuracy)가 어떻게 변하는지 계속 기록해달라." (휴게소나 주유소를 얼마나 지나는지
     확인)

> 이 단계에서는 아직 출발(학습)하지 않았습니다. 어떤 목적지를 어떤 방법으로 갈지 계획만 세운 상태입니다.


2. model.fit(): '안내 시작' 버튼 누르기

이 단계는 설정된 규칙에 따라 실제 데이터로 학습을 '시작'하고 '실행'하는 과정입니다.
```
model.fit(X, YT, epochs=999)
```
* `X`, `YT` (데이터): 운전을 해야 할 실제 '지도'와 '도로망' 데이터를 제공합니다.
   * X: 문제지 (입력 데이터)
   * YT: 정답지 (실제 값)


* `epochs=999` (에폭): 운전을 얼마나 반복할지 결정합니다.
   * "우리가 가진 전체 지도(데이터셋)를 총 999번 반복해서 운전(학습)하면서 최적의 경로를 찾아라."


> 이 명령이 실행되는 순간, 모델은 compile에서 설정한 'mse'를 최소화하기 위해 'sgd' 방식으로 X와 YT
데이터를 999번 반복 학습하며 내부의 가중치와 편향을 실제로 업데이트하기 시작합니다.

| 함수 | 역할 | 비유 |
| :--- | :--- | :--- |
| **`model.compile()`** | 학습 **규칙/전략** 설정 | 내비게이션 **목적지 및 경로 설정** |
| **`model.fit()`** | 실제 데이터로 학습 **시작/실행** | **'안내 시작'** 버튼 누르기 |

---
### Keras 옵티마이저와 학습률(Learning Rate) 설정

- model.compile() 시 optimizer를 설정할 때, 학습률(lr)을 직접 지정하지 않아도 코드가 동작하는 경우가
있습니다. 이는 케라스(Keras)가 사용자를 위해 기본값(default)을 자동으로 적용해주기 때문입니다.

- 사용자가 모든 세부사항을 지정하지 않아도 코드가 돌아갈 수 있도록, 대부분의 라이브러리는 중요한 파라미터에
대해 합리적인 기본값을 가지고 있습니다.


1. 암시적 방법

```python
model.compile(optimizer='sgd', loss='mse')
```
- 케라스는 '가장 기본적인 SGD 옵티마이저를 기본 설정값으로 사용하라'는 의미로 받아들입니다.

> tf.keras.optimizers.SGD의 기본 학습률은 `0.01` 입니다.
> 따라서 위 코드는 아래와 같이 학습률을 직접 0.01로 지정한 것과 동일하게 동작합니다.


```python
# optimizer='sgd' 라고 쓴 것은 아래 코드와 같습니다.
from tensorflow.keras.optimizers import SGD

optimizer_with_defaults = SGD(learning_rate=0.01)

model.compile(optimizer=optimizer_with_defaults, loss='mse')
```


2. 명시적 방법(학습률을 직접 설정하는 방법):
- 만약 다른 학습률(예: 0.05)을 사용하고 싶다면, 옵티마이저 객체를 직접 만들어서 compile 함수에 전달해야합니다.

```python
from tensorflow.keras.optimizers import SGD

# 1. 원하는 학습률로 옵티마이저 객체를 직접 생성
my_optimizer = SGD(learning_rate=0.05)

# 2. compile 할 때 문자열 대신 생성한 객체를 전달
model.compile(optimizer=my_optimizer, loss='mse')
```
---

## 1입력 1출력 인공 신경망

<img width="937" height="277" alt="image" src="https://github.com/user-attachments/assets/b37f0d30-e42a-4f0b-a462-7242fc642e4c" />

<img width="618" height="173" alt="image" src="https://github.com/user-attachments/assets/eed90e86-2c69-4d7c-9dd9-bf61ada3b62e" />

```python
import tensorflow as tf
import numpy as np

X = np.array([[2]])
YT = np.array([[10]])
W = np.array([[3]])
B = np.array([1])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
    ])

model.layers[0].set_weights([W,B])

model.compile(optimizer='sgd', loss='mse')
Y = model.predict(X)
print(Y)


model.fit(X,YT,epochs=999,verbose=0)
print('W=', model.layers[0].get_weights()[0])
print('B=', model.layers[0].get_weights()[1])

Y= model.predict(X)
print(Y)

```
```
model.fit(X,YT,epochs=999,verbose=0)
```
* `verbose=0`: 아무것도 안 보여줌
* `verbose=1`: 진행 막대 보여줌 (기본값)
* `verbose=2`: 에폭마다 한 줄씩 요약해서 보여줌


## 2입력 1출력

<img width="917" height="312" alt="image" src="https://github.com/user-attachments/assets/3b2c4ac9-c7a4-4189-b1a4-20f267c31a83" />

```python
X=np.array([[2,3]]) # 입력데이터
YT=np.array([[27]]) # 목표데이터(라벨)
W=np.array([[3],[4]]) # 가중치
B=np.array([1]) # 편향
```
  
---
## Keras에서 특정 조건 도달 시 학습 조기 종료하기 (Early Stopping)

model.fit()에서 epochs를 큰 값으로 설정하더라도, 특정 조건을 만족하면 학습을 그 전에 멈추게 하여 시간과 자원을 효율적으로 사용할 수 있습니다.

이러한 기능은 콜백(Callback)을 통해 구현하며, 특히 사용자 정의 콜백(Custom Callback)을 사용하면 원하는거의 모든 규칙을 만들 수 있습니다.

#### 특정 손실(Loss) 값 도달 시 학습 멈추기
학습 손실(loss) 값이 특정 목표치(예: 1e-6 또는 0.000001) 이하로 떨어졌을 때 학습을 멈추는 방법입니다.

1. 사용자 정의 콜백 클래스 만들기

먼저, tf.keras.callbacks.Callback을 상속받아 우리만의 규칙을 가진 클래스를 정의합니다.

```python

1 import tensorflow as tf
    
3# 목표 손실 값에 도달하면 학습을 멈추는 콜백 정의
class StopAtTargetLoss(tf.keras.callbacks.Callback):
"""
학습 중 loss가 특정 목표값 이하로 떨어지면 학습을 중지시키는 콜백.
"""
  def on_epoch_end(self, epoch, logs=None):
    # logs 딕셔너리에서 현재 에폭의 loss 값을 가져옴
    current_loss = logs.get('loss')
    
    # 목표 손실 값 (예: 0.000001)
    target_loss = 1e-6

  # loss 값이 존재하고, 목표치보다 작거나 같으면 학습 중지
  if current_loss is not None and current_loss <= target_loss:
    print(f"\n에폭 {epoch+1}: 손실 값이 {current_loss:.7f} (목표치: {target_loss})에
  도달하여 학습을 조기 종료합니다.")
    self.model.stop_training = True
```

* on_epoch_end: 각 에폭이 끝날 때마다 Keras에 의해 자동으로 호출되는 함수입니다.
* logs.get('loss'): 현재 에폭의 학습 손실(loss) 값을 가져옵니다.
* self.model.stop_training = True: 이 코드가 실행되면 model.fit()은 다음 에폭으로 넘어가지 않고 즉시
 학습을 멈춥니다.

2. model.fit()에 콜백 적용하기
위에서 만든 콜백을 model.fit() 함수의 callbacks 인자에 리스트 형태로 전달
```python
# 1. 위에서 정의한 콜백의 인스턴스(객체) 생성
my_callback = StopAtTargetLoss()

# 2. model.fit 호출 시 callbacks 리스트에 추가
# verbose=1로 설정하여 출력을 확인하는 것이 좋습니다.
history = model.fit(X_train,
          y_train,
          epochs=999,
          verbose=1,
          callbacks=[my_callback])
```
모델은 999번의 에폭을 모두 채우기 전에, 학습 손실(loss)이 0.000001 이하로 떨어지는 순간 "학습을 조기  종료합니다"라는 메시지를 출력하며 자동으로 멈추게 됩니다.

---
#### 일반적인 EarlyStopping 콜백 (과적합 방지용)

Keras에는 EarlyStopping이라는 기본 콜백도 있습니다. 이것은 특정 값이 '개선되지 않을 때' 멈추는 용도이며, 과적합(Overfitting)을 방지하기 위해 더 일반적으로 사용됩니다.

1. 검증 데이터가 없을떄
```python
# monitor를 'val_loss'가 아닌 'loss'로 변경
# patience는 너무 길면 오래 걸리니 10 정도로 조정하는 것을 추천
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# validation_data 인자를 제거
model.fit(X, YT,
          epochs=1500,
          verbose=0,
          callbacks=[early_stopping])
```

2. 검증 데이터가 있을때
```python

# 예시: 검증 손실(val_loss)이 5번의 에폭 동안 개선되지 않으면 학습 중지
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# validation_data를 함께 제공해야 monitor='val_loss'를 사용할 수 있음
model.fit(X_train, y_train,
          epochs=999,
          validation_data=(X_val, Y_val),
          callbacks=[early_stopping])
```
>'특정 목표값'에 도달했을 때 멈추고 싶다면, 위에서 설명드린 사용자 정의 콜백을 만드는다.

---
## 2입력 2은닉 2출력

<img width="816" height="334" alt="image" src="https://github.com/user-attachments/assets/9e51b341-ffcc-41e2-9623-738610bb9840" />

<img width="792" height="345" alt="image" src="https://github.com/user-attachments/assets/ac457df4-86d4-4539-b69f-ec13e65fa342" />

```python
import tensorflow as tf
import numpy as np

X = np.array([[.05, .10]])
YT = np.array([[.01, .99]])
W = np.array([[.15, .25], [.20, .30]])
B = np.array([.35, .35])
W2 = np.array([[.40, .50],[.45, .50]])
B2 = np.array([.60, .60])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2), # 은닉층 h1 ,h2
    tf.keras.layers.Dense(2) # 출렬층 y1, y2
    ])


model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer='sgd', loss='mse')
Y = model.predict(X)
print(Y)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',  min_delta=1e-7,.patience=5) # 0.0000001. 이 값보다 작은 변화는 '개선'으로 치지 않음

# validation_data를 함께 제공해야 monitor='val_loss'를 사용할 수 있음
model.fit(X,YT,
          epochs=1500,
          verbose=1,          
          callbacks=[early_stopping])


#model.fit(X,YT,epochs=2000,verbose=2)
print('W=', model.layers[0].get_weights()[0])
print('B=', model.layers[0].get_weights()[1])
print('W2=', model.layers[1].get_weights()[0])
print('B2=', model.layers[1].get_weights()[1])

Y= model.predict(X)
print('Y=',Y)

```

---

## w, b, E 관계
```python
np.random.uniform(-200, 200, 10000)
```
매우 넓은 범위(-200 ~ 200)에 걸쳐 수많은 경우의   수(10,000개)를 무작위로 샘플링

<img width="828" height="60" alt="image" src="https://github.com/user-attachments/assets/97166a42-1e92-41a0-a6eb-2d2ceeaf34d4" />

- w = np.random.uniform(-2, 2, 4)
- b = np.random.uniform(-2, 2, 4)
> -2이상 2미만 범위에서 4개의 값 샘플링

#### basic
```python
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
ax.set_title("wbE", size = 20)

ax.set_xlabel("w", size = 14)
ax.set_xlabel("b", size = 14)
ax.set_xlabel("E", size = 14)

x = 2
yT = 10

w = np.random.uniform(-200, 200, 10000)
b = np.random.uniform(-200, 200, 10000)

y = (x*w) + b
E = ((y-yT)**2) / 2

ax.plot(w, b, E, 'g.')
plt.show()
```

<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/d34ad0f9-ba83-4063-90bf-a4c7ecb3fcb8" />

####  학습 과정 살펴보기

```python
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
ax.set_title("wbE", size = 20)

ax.set_xlabel("w", size = 14)
ax.set_xlabel("b", size = 14)
ax.set_xlabel("E", size = 14)

x = 2
yT = 10

w = np.random.uniform(2, 7, 10000)
b = np.random.uniform(0, 4, 10000)

y = (x*w) + b
E = ((y-yT)**2) / 2

ax.plot(w, b, E, 'g.')

x = 2
w = 3
b = 1
yT = 10
lr = 0.01

wbEs = []
EPOCHS = 200

for epoch in range(EPOCHS):
        y = x*w + 1*b
        E = (y - yT)**2 /2
        yE = y - yT
        wE = yE * x
        bE = yE *1
        w -= lr*wE
        b -= lr*bE
        
        wbEs.append(np.array([w,b,E]))

data = np.array(wbEs).T
line, = ax.plot([],[],[],'r')

def animate(epoch, data, line):
    print(epoch, data[2, epoch])
    line.set_data(data[:2, :epoch])
    line.set_3d_properties(data[2,:epoch])

from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, animate, EPOCHS, fargs=(data, line), interval=20000/EPOCHS)


plt.show()

```

<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/d2500e8b-abda-42f5-8b2c-c5ce36edb58c" />

<img width="500" height="339" alt="image" src="https://github.com/user-attachments/assets/d6bc3e4b-dc61-4d2e-a818-694508eece95" />


- 학습에 따라 오차율이 점점 줄어드는걸 확인 할 수 있다.

---

## 활성화 함수
활성화 함수는 '뉴런 하나'가 어떻게 반응할지를 결정하고, 옵티마이저는 '모델 전체'가 어떻게 더 똑똑해질지를 결정합니다.

1.  Sigmoid : 
- 입력값이 아무리 크더라도 출력값의 범위가 0에서 1사이로 매우 작아 신경망을 거칠수록 출력값은 점점더 작아져 0에 수렴하게 됩니다.

<img width="858" height="207" alt="image" src="https://github.com/user-attachments/assets/846a5842-55c8-4d2b-a4d3-89468acf33d7" />

---
2.  ReLU(Rectified Linear Unit)
- ReLU 함수의 경우 입력값이 음수가 될 경우 출력이 0이 되기 때문에 이런 경우에는 어떤 노드를 완전히 죽게 하여 어떤 것도 학습하지 않게 합니다.

<img width="823" height="243" alt="image" src="https://github.com/user-attachments/assets/dc412ab1-c28b-44a0-b58b-950694772a52" />

---
3.  Softmax
- 소프트맥스 활성화 함수는 시그모이드 활성화 함수가 더욱 일반화된 형태
- 시그모이드 함수와 비슷하게 이것은 0에서 1사이의 값들을 생성합니다. 소프트맥스 함수는 은닉층에서는 사용하지 않으며, 다중 분류(classification) 모델에서 출력층에서만 사용합니다.

<img width="477" height="259" alt="image" src="https://github.com/user-attachments/assets/a2d33418-f6ec-48e5-bff9-50160191e2d9" />

**일반적으로 은닉층에는 ReLU 함수를 적용하고, 출력층은 시그모이드 함수나 소프트맥스 함수를 적용합니다.**

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return x*(x>0)

def step(x):
    return x>0

fig, axes = plt.subplots(3, 1, figsize=(8, 12)) # fig는 전체 창, axes는 각 그래프를 그릴 3개의 구역(도화지) 3행 1열
fig.suptitle("Activation Functions", size=20) # 전체 창의 제목

x_sigmoid = np.random.uniform(-10, 10, 1000)
y_sigmoid = sigmoid(x_sigmoid)
axes[0].plot(x_sigmoid, y_sigmoid, 'r.')
axes[0].set_title("Sigmoid Function")
axes[0].grid(True)

x_relu = np.random.uniform(-10, 10, 1000)
y_relu = ReLU(x_relu)
axes[1].plot(x_relu, y_relu, 'g.')
axes[1].set_title("ReLU Function")
axes[1].grid(True)

x_step  = np.random.uniform(-10, 10, 1000)
y_step  = step (x_step )
axes[2].plot(x_step , y_step , 'b.')
axes[2].set_title("Step  Function")
axes[2].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

```
<img width="1910" height="1022" alt="image" src="https://github.com/user-attachments/assets/13b08218-69ab-4049-9541-73fec0c27270" />

```
np.exp(x)
```
---
#### exp()
- NumPy 라이브러리의 exp() 함수는 입력받은 숫자 또는 배열 x의 모든 요소에 대해 eˣ (e의 x제곱) 값을 계산해줍니다.

그래서 어디에 쓰이나요? → 소프트맥스(Softmax) 함수

이 두 가지 마법은 분류 모델의 마지막 단계인 소프트맥스(Softmax) 함수에서 완벽하게 활용됩니다.


상황: 모델이 '개', '고양이', '새' 사진을 보고 각각의 점수를 [개=1, 고양이=3, 새=0.5] 라고 예측했습니다.
우리는 이 점수를 '확률'로 바꾸고 싶습니다.


 1. `exp()` 적용 (마법 1 + 마법 2)
     * 점수 [1, 3, 0.5]에 exp()를 적용해서 모두 양수로 만들고, 가장 점수가 높은 '고양이'를 확실한
       주인공으로 부각시킵니다.
     * exp([1, 3, 0.5]) => [2.7, 20.1, 1.6]


 2. 전체 합으로 나누기 (정규화)
     * 위에서 나온 값들을 모두 더합니다: 2.7 + 20.1 + 1.6 = 24.4
     * 각각의 값을 전체 합으로 나누어, 총합이 1인 확률로 만듭니다.
     * [2.7/24.4, 20.1/24.4, 1.6/24.4] => [0.11, 0.82, 0.07]


최종 결과: 모델은 이 사진이 '개'일 확률 11%, '고양이'일 확률 82%, '새'일 확률 7%라고 최종 결론을
내립니다.


결론적으로 np.exp()는 어떤 숫자든 양수로 바꾸고, 그중 가장 큰 값을 더 특별하게 강조하는 두 가지 강력한
특징 때문에, 여러 점수들을 명확한 '확률'로 바꿔야 하는 머신러닝 분류 문제에서 필수적인 함수로 사용됩니다

---

## activivation function

<img width="595" height="334" alt="image" src="https://github.com/user-attachments/assets/0a89537f-fc8c-48ce-ab9c-cbca3c32101a" />

```python
from math import exp
x1, x2 = 0.05, 0.10
w1, w2 = 0.15, 0.20
w3, w4 = 0.25, 0.30
w5, w6 = 0.40, 0.45
w7, w8 = 0.50, 0.55
b1, b2, b3, b4 = 0.35, 0.35, 0.60, 0.60
y1T, y2T= 0.01, 0.99
lr = 0.01

EPOCH = 1000

for epoch in range(EPOCH):
    
    h1 = (x1*w1) + (x2*w2) + (1*b1)
    h2 = (x1*w3) + (x2*w4) + (1*b2)
    
    #ReLU feed forward
    h1 = h1 if h1 > 0 else 0
    h2 = h2 if h2 > 0 else 0
    
    
    y1 = (h1*w5) + (h2*w6) + (1*b3)
    y2 = (h1*w7) + (h2*w8) + (1*b4)
    
    #sigmoid feed forward
    y1 = 1 / (1 + exp(-y1))
    y2 = 1 / (1 + exp(-y2))
    
    E = ((y1 - y1T)**2) / 2  + ((y2 - y2T)**2) / 2
    
    y1E = y1 - y1T
    y2E = y2 - y2T
    
    # sigmoid back propagation
    
    y1E = y1 * (1-y1) * y1E
    y2e = y2 * (1-y2) * y2E
    
    w5E = y1E*h1
    w6E = y1E*h2
    w7E = y2E*h1
    w8E = y2E*h2
    b3E = y1E*1
    b4E = y2E*1
    
    h1E = (y1E*w5) + (y2E*w7) 
    h2E = (y1E*w6) + (y1E*w8)
    
    # ReLU back propagation
    h1E = h1E if h1 > 0 else 0
    h2E = h2E if h2 > 0 else 0


    w1E = h1E*x1
    w2E = h1E*x2
    w3E = h1E*x1
    w4E = h1E*x2
    b1E = h1E*1
    b2E = h2E*1
    

    
    w5 -= lr*w5E
    w6 -= lr*w6E
    w7 -= lr*w7E
    w8 -= lr*w8E
    b3 -= lr*b3E
    b4 -= lr*b4E
    
    w1 -= lr*w1E
    w2 -= lr*w2E
    w3 -= lr*w3E
    w4 -= lr*w4E
    b1 -= lr*b1E
    b2 -= lr*b2E
    
    
    if epoch % 100 ==99:
        print(f'epoch = {epoch}')
        print(f' y1 : {y1:.6f}')
        print(f' y2 : {y2:.6f}')
    if E < 0.0000001:
        break;
    

print(f'w1,w3 = {w1:.6f},{w3:.6f}')
print(f'w2,w4 = {w2:.6f},{w4:.6f}')
print(f'b1,b2 = {b1:.6f},{b2:.6f}')
print(f'w5,w7 = {w5:.6f},{w7:.6f}')
print(f'w6,w8 = {w6:.6f},{w8:.6f}')
print(f'b3,b4 = {b3:.6f},{b4:.6f}')
```

<img width="417" height="170" alt="image" src="https://github.com/user-attachments/assets/22f054f2-8c50-48a4-88f5-8958cb3efa11" />

---

#### tensorflow에 activation 적용

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu'), # 은닉층
    tf.keras.layers.Dense(2, activation='sigmoid') # 출력층
    
    ])
```

<img width="736" height="253" alt="image" src="https://github.com/user-attachments/assets/262d7d8b-8d7f-4443-920c-00a3a0dac21b" />

---
## 선형함수 
#### 출력층에 linear 함수 적용

- 출력층에 linear 함수를 적용하면 출력단의 값을 그대로 내보낸다.


```
model=tf.keras.Sequential([ tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear') # 출력층 linear 함수 적용
]) 
```

<img width="745" height="225" alt="image" src="https://github.com/user-attachments/assets/9f087935-b82e-4256-a2cc-b44456a178f3" />

- epoch = 571 종료
---
#### 선형함수에 대한 질문
1. 선형 함수를 쓰는이유
2. default 랑 linear 함수랑 같은가?
3. 출력층의 활성화 함수에 따라 학습 속도가 달라지는 이유

https://github.com/hjjk2688/AI/blob/main/activation_function.md

---

## Softmax & cross entropy
