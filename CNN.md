# CNN

## CNN(Convolution Neural Network)

<img width="836" height="395" alt="image" src="https://github.com/user-attachments/assets/98cba750-17a6-4f00-9b13-49b881ad8b24" />

- kernel = filter(직육면체 가중치)
- 기본 데이터 형태: 가로x세로x깊이 
- 각 특징을 추출하는 filter(convolution)는 특징의 수만큼 있어야된다. ( 각 특징을 추출(곱해서) 점을 찍어 새로운 이미지를 생성)
- Pooling: 그림의 크기를 1/4로 줄이는작업(연산을 줄이기위해서, 가장 큰 특징을 선택)

---

### 깊이 의미
입력 이미지의 깊이(Depth)는 색상 채널(Color Channel)의 개수를 의미합니다.

이미지 데이터는 보통 (가로 크기, 세로 크기, 깊이)의 3차원 형태로 표현된다.

1. 흑백 이미지 (Grayscale Image)
    - 흑백 이미지는 오직 밝기 정보만 가지고 있습니다 (밝다, 어둡다).
    - 따라서 채널이 1개입니다.
    - 이 경우, 깊이(Depth) = 1 입니다.
    - 예: MNIST 데이터셋의 이미지는 (28, 28, 1) 형태로, 깊이가 1입니다.

2. 컬러 이미지 (Color Image)
    - 컬러 이미지는 일반적으로 빨강(Red), 초록(Green), 파랑(Blue) 세 가지 색상의 조합으로 표현됩니다. 이를 RGB
    채널이라고 합니다.
    - 각 색상 채널이 별도의 2D 이미지처럼 존재하고, 이 세 개가 겹쳐져서 하나의 컬러 이미지를 만듭니다.
    - 따라서 채널이 3개입니다.
    - 이 경우, 깊이(Depth) = 3 입니다.
    - 예: CIFAR-10 데이터셋의 이미지는 (32, 32, 3) 형태로, 깊이가 3입니다.

---
###  CNN 하이퍼파라미터: 필터 개수 결정 원리

합성곱 신경망(CNN)에서 필터(Filter)의 개수(예: 32개, 64개)를 설정하는 것은 **개발자(설계자)의 역할**이며, 이는 딥러닝에서 중요한 **하이퍼파라미터(Hyperparameter)**에 해당합니다.

#### 1. 필터 개수의 의미 (출력 깊이)

| 요소 | 설명 | 개발자 결정 여부 |
| :--- | :--- | :--- |
| **필터 개수** | 해당 단계에서 이미지의 특징을 **몇 가지 관점**으로 분석할 것인지를 결정 | **✅ 개발자 선택** |
| **출력 깊이** | 필터 개수와 동일하며, 생성되는 **특징 맵(Feature Map)의 개수** | **✅ 개발자 선택** |
| **필터 깊이** | **직전 입력 이미지의 깊이**와 반드시 일치해야 함 | **❌ 수학적 규칙** |

**예시:** 1차 합성곱에서 필터 개수를 32개로 설정했다는 것은, 입력 이미지에서 **32가지 종류의 특징**을 추출하겠다는 의미입니다.

#### 2. 필터 개수 설정의 일반적인 전략

| 단계 | 개수 (예시) | 전략 및 이유 |
| :--- | :--- | :--- |
| **초기 단계** (1차 합성곱) | 32개 | 단순한 특징(Low-level Feature)을 추출하기 때문에 적은 수로 시작 |
| **후기 단계** (2차 합성곱 이후) | 64개, 128개 | 초기 특징들을 조합하여 **복잡하고 추상적인 특징**을 추출하기 위해 개수를 늘려 정보의 풍부도를 높임 |
| **기술적 선택** | $2^n$ (32, 64, 128...) | 컴퓨터 하드웨어(특히 GPU)가 **2의 거듭제곱** 형태의 데이터 처리에 가장 효율적이기 때문에 관례적으로 사용 |

---

## CNN의 일반적인 처리 흐름 (Convolutional Neural Network Flow)

### 1. 특징 추출 (Feature Extraction)
#### ▶ Convolutional Layers (+ Pooling Layers)

- 이 단계에서 이미지의 다양한 특징을 **추출**한다.
- **Convolutional Layer**는 필터(Filter)를 이용해 다음과 같은 특징을 단계적으로 학습한다:
  - 저수준 특징: 선(Line), 모서리(Edge), 질감(Texture)
  - 고수준 특징: 눈, 코, 귀 등의 형태
- 이 과정을 통해 생성되는 출력은 **Feature Map(특징 맵)** 이다.
- Pooling Layer는 Feature Map의 공간 크기를 줄여 **연산량 감소** 및 **특징의 요약** 역할을 한다.

---

### 2. 분류 (Classification)
#### ▶ Fully Connected Layers (FC Layers)

- Conv Layers에서 생성된 Feature Map을 **Flatten**(1차원 벡터로 펼치기)하여 FC Layer에 전달한다.
- 중요한 점:
  - **학습은 Conv Layer와 FC Layer 전체에서 동시에 일어난다.**
- FC Layer의 역할:
  - "앞 단계에서 추출된 특징들의 조합이 어떤 클래스인지"를 종합적으로 판단하는 규칙을 학습한다.

---

### 3. 출력 (Output)
#### ▶ Softmax Function

- FC Layer의 최종 출력값인 **Logits**을 Softmax 함수에 통과시켜 클래스별 **확률**로 변환한다.
- 예:
  - 고양이 85%
  - 개 10%
  - 새 5%

---

### 📌 전체 처리 흐름 요약.
```text
[입력 이미지]
↓
[Convolution + Pooling Layers — 특징 추출]
↓
[Flatten — 1차원 벡터 변환]
↓
[Fully Connected Layers — 특징 종합 및 분류]
↓
[Softmax — 클래스 확률 출력]
↓
[최종 예측 결과]
```
---

#### ✔ 학습(Training)은 전체 계층에서 동시에 일어난다
- Loss(예측값 vs 정답)의 오차를 줄이기 위해
- **모든 Conv Layer + 모든 FC Layer의 가중치가 역전파(Backpropagation)에 의해 함께 업데이트됨**

---

## 구현

<img width="1000" height="547" alt="image" src="https://github.com/user-attachments/assets/ae88708a-0242-4e4d-a3ee-7e5471c7bb03" />

### CNN 신경망 구조 단계별 요약

**➊ 입력 층 (Input)**
- **크기:** 28x28x1 (세로 x 가로 x 깊이)
- **특징:** 깊이(1)는 채널(Channel)을 의미함

**➋ 필터 정의 (Filter Setup)**
- **규칙:** 필터의 깊이는 **입력 이미지의 깊이**와 반드시 같아야 함
- **적용:** 입력 깊이가 1이므로, **3x3x1** 크기의 필터가 적용됨

**➌ 1차 합성곱 (1st Convolution)**
- **규칙:** 필터의 개수는 **출력 이미지의 깊이**와 같아야 함
- **구성:** 출력 깊이가 32이므로, **32개의 필터** 필요 (3x3x1x32)
- **결과:** **28x28x32** 크기의 특징 맵 생성

**➍ 2차 필터 정의**
- **규칙:** 이전 단계의 출력 깊이가 32이므로 필터 깊이도 커져야 함
- **적용:** **3x3x32** 크기의 필터가 적용됨

**➎ 2차 합성곱 (2nd Convolution)**
- **구성:** 출력 깊이가 64이므로, **64개의 필터** 필요 (3x3x32x64)
- **결과:** **28x28x64** 크기의 특징 맵 생성

**➏ 모으기 (Max Pooling)**
- **역할:** 이미지의 가로, 세로 크기를 축소 (보통 1/2로 줄임)
- **결과:** 28x28 → **14x14** (깊이 64는 유지되어 **14x14x64**)

**➐ 평탄화 (Flatten)**
- **역할:** 3차원 입체 데이터를 1차원 배열로 변환
- **계산:** 14 x 14 x 64 = **12,544**
- **결과:** 12,544개의 노드가 되어 이후 완전 연결 계층(Dense Layer)으로 전달

---

## Convlution 

- convlution(합성곱)
- 합성곱: 항볼별 곱을 다 더한 값
- CNN의 합성곱(Convolution)은 일반적인 행렬 곱셈(Matrix Multiplication)이 아님
- "각 자리의 값끼리 곱한 다음 모두 더하는" 방식 (Element-wise Multiplication and Sum)

#### 1. 3x3 입력: filter size

<img width="800" height="428" alt="image" src="https://github.com/user-attachments/assets/637d52f7-f3f7-4568-85ac-05fc8898bea2" />

```python
import numpy as np

np.random.seed(1)
image = np.random.randint(5, size=(3,3))
print('image = \n', image)

filter = np.random.randint(5, size=(3,3))
print('filter = \n', filter)

image_x_filter = image * filter
print('image_x_filter = \n', image_x_filter)

convolution = np.sum(image_x_filter)
print('convolution = \n', convolution)

```
<img width="258" height="262" alt="image" src="https://github.com/user-attachments/assets/c00d2d97-814a-4468-9262-3bd6f8d1fd33" />

#### 2.  4x4 입력 이미지와 3x3 필터에 대해 합성 곱
- 출력이 줄어든다. (2x2)

<table>
    <tr>
        <td><img width="792" height="514" alt="image" src="https://github.com/user-attachments/assets/1facdab7-12d6-42ec-8bb9-ef26722f2bb5" /></td>
        <td><img width="773" height="345" alt="image" src="https://github.com/user-attachments/assets/ecd9b24e-5d78-4b09-93d2-e34ea021e25b" /></td>
        <td><img width="830" height="814" alt="image" src="https://github.com/user-attachments/assets/d19bc944-0b59-4e1b-b8c8-905bcaf0525c" /> </td>
    </tr>    
</table>

```python
import numpy as np

np.random.seed(1)
image = np.random.randint(5, size=(4,4))
print('image = \n', image)

filter = np.random.randint(5, size=(3,3))
print('filter = \n', filter)

convolution = np.zeros((2,2))

for row in range(2):
    for col in range(2):
        window = image[row:row+3, col:col+3]
        print('window(%d, %d) =\n' %(row,col), window)
        print('window(%d, %d)*filter =\n' %(row,col), window*filter)
        convolution[row,col] = np.sum(window*filter)
print('convolution =\n', convolution)

```

<table>
    <tr>
        <td><img width="201" height="170" alt="image" src="https://github.com/user-attachments/assets/f404cc6b-cff3-4280-a8bd-747ae952fc52" /></td>
        <td><img width="259" height="200" alt="image" src="https://github.com/user-attachments/assets/2f1d17ff-c2c6-47b1-a544-b9a6d45da592" /></td>
        <td><img width="204" height="77" alt="image" src="https://github.com/user-attachments/assets/86574bd5-470a-4580-a70d-15f2c4c8dd93" /></td>
    </tr>
</table>

#### stride 란 ?
- stride(스트라이드): 필터(Filter)가 이미지 위를 이동하는 보폭 또는 간격
- 필터가 한 번의 합성곱 연산을 마친 후, 다음 연산을 위해 얼마나 이동할지를 결정하는 값

**stride를 사용하는 이유**

1. 출력 크기 조절 (Downsampling)
    - 스트라이드 값을 1보다 크게 설정하면 출력 피처 맵의 크기를 효과적으로 줄일 수 있습니다.(다운샘플링(Downsampling))

2. 계산량 감소
    - 출력 맵의 크기가 작아지므로, 다음 레이어에서 처리해야 할 계산량이 줄어들어 모델의 전체적인 연산속도가 빨라집니다.

3. 풀링(Pooling) 레이어 대체
---

## Padding
- 합성곱을 수행하면 출력 이미지는 입력이미지보다 작아지게된다 ( image와 filter size가 같이않을떄)
- padding: 입력 이미지의 크기와 출력 이미지의 크기를 같게 하기 위해서는 입력 이미지의 크기를 늘려 주는 과정
- 맨위에 CNN에서는 28X28X1 입력을 padding하고 3X3X1X32 filter와 계산 해서 28X28X32 convlution 출력을만듬

<img width="762" height="395" alt="image" src="https://github.com/user-attachments/assets/6c218ffe-d2cc-46e3-91ee-a38ad5bbe883" />

#### 출력 이미지 크기 
$$\left\lfloor \frac{N+2P-K}{S} \right\rfloor + 1 \times \left\lfloor \frac{N+2P-K}{S} \right\rfloor + 1$$

```
- 입력 이미지 크리 = N * N
- 필터 크기 = K * K
- padding = P
- stride = S

```
- 예시

$$\left\lfloor \frac{4+2\times1-3}{1} \right\rfloor + 1 \times \left\lfloor \frac{4+2\times1-3}{1} \right\rfloor + 1 = 4\times4$$

```
- 입력 이미지 크리 = 4 * 4
- 필터 크기 = 3 * 3
- padding = 1
- stride = 1
```
 => 출력 이미지 크기 = 4 X 4

```python
import numpy as np

np.random.seed(1)
image = np.random.randint(5, size=(4,4))
print('image = \n', image)

filter = np.random.randint(5, size=(3,3))
print('filter = \n', filter)

image_pad = np.pad(image,((1,1),(1,1)))
print('image_pad =\n', image_pad)

convolution = np.zeros((4,4))

for row in range(4):
    for col in range(4):
        window = image_pad[row:row+3, col:col+3]
        convolution[row,col] = np.sum(window*filter)

print('convolution =\n', convolution)

```

<img width="530" height="438" alt="image" src="https://github.com/user-attachments/assets/03f3fad3-daa9-4996-9191-0d3a0cf824dc" />

---

## Pooling
특정 특징값을 뽑아서(요약해서) 전체 데이터의 사이즈를 줄이는 것
-  풀링은 Convolution Layer를 통해 나온 피처 맵(Feature Map)에 대해 처리한다.

**풀링(Pooling)의 두 가지 주요 방식**

1. 최대 풀링 (Max Pooling) - 가장 널리 사용
   * 정해진 영역(예: 2x2)에서 가장 큰(가장 두드러진) 특징값 하나만을 선택합니다.
   * 의미: 그 영역에서 가장 '활성화된' 또는 '중요한' 특징이 무엇인지만 남기고 나머지는 버리는 것입니다.
     이를 통해 중요한 특징을 유지하면서 크기를 줄입니다.


2. 평균 풀링 (Average Pooling)
   * 정해진 영역에 있는 모든 특징값들의 평균을 계산하여 대표값으로 사용합니다.
   * 의미: 특징들을 전반적으로 부드럽게 요약하는 효과가 있습니다.

| 구분 | Max Pooling | Average Pooling |
| :--- | :--- | :--- |
| **방식** | 영역 내 **최대값** 추출 | 영역 내 **평균값** 계산 |
| **효과** | 가장 두드러진, 날카로운 특징 강조 | 전반적인 특징을 부드럽게 요약 |
| **주요 사용처** | CNN의 중간 레이어 (특징 추출) | **CNN의 마지막 단 (Global Average Pooling)** |
| **장점** | 중요한 특징 보존에 유리 | 노이즈에 덜 민감, 전반적인 분포 반영 |

Max Pooling: 네트워크의 **중간 단계**에서 특징을 잘 보존하여 주로 사용됩니다.
Average Pooling (Global Average Pooling): **네트워크의 마지막**에서 모델을 효율적으로 만들고 전반적인 특징 분포를 요약하기 위해 매우 중요하게 사용됩니다.  

**폴링 효과**
1. 데이터 크기 감소 (Downsampling)
   * 피처 맵의 가로, 세로 크기를 줄여줍니다. (예: 2x2 풀링, 스트라이드 2 -> 크기가 절반으로 줄어듦)
   * 이로 인해 모델의 전체적인 계산량(연산 속도)이 크게 줄어듭니다.


2. 과적합(Overfitting) 방지
   * 데이터의 크기를 줄이면서 약간의 위치 변화나 노이즈에도 모델이 덜 민감하게 반응하도록 만듭니다.
   * 즉, 너무 세세한 부분에 집착하지 않고 전반적인 특징을 보도록 유도하여 과적합을 억제하는 효과가
     있습니다.


3. 주요 특징 강조 (특히 Max Pooling의 경우)
   * Max Pooling은 가장 강한 신호만 남기므로, 해당 영역의 핵심 특징을 더욱 강조하는 효과가 있습니다.
  
#### Max pooling

<img width="471" height="286" alt="image" src="https://github.com/user-attachments/assets/a0ed6c3b-9907-4b98-b28c-030c0d139675" />

```python
--- 풀링(Pooling) 코드 같음---

max_pooled = np.zeros((2,2))

for row in range(0,2):
    for col in range(0,2):
        window = convolution[2*row:2*row+2, 2*col:2*col+2]
        max_pooled[row,col] = np.max(window)
print('max_pooled = \n', max_pooled)
```

<table>
    <tr>
        <td><img width="343" height="320" alt="image" src="https://github.com/user-attachments/assets/efa19e79-cbf5-4fa3-885d-02b9fd449089" /></td>
        <td><img width="298" height="193" alt="image" src="https://github.com/user-attachments/assets/579ac2ad-8e15-42d0-be11-c13f0896f02a" /></td>        
    </tr>    
</table>


#### pooling size 2x2를 가장 많이 사용할까? (pool_size=(2, 2) 와 strides=2 )

1. 적절한 다운샘플링(Downsampling)
   * 이 설정은 피처 맵의 가로, 세로 크기를 정확히 절반으로 줄이는 효과가 있습니다. (예: 28x28 -> 14x14)
   * 너무 많은 정보를 한 번에 잃지 않으면서도, 계산량을 효과적으로 줄여줍니다.

2. 균형 잡힌 정보 손실
   * 2x2 영역(4픽셀)에서 가장 중요한 특징 1개만 남기는 것은, 정보를 요약하되 너무 많은 디테일을 잃지 않는 가장 균형 잡힌 선택으로 여겨집니다.

* `2x2`: (강력 추천, 표준) 가장 일반적이고 안정적인 선택입니다.
* `3x3`: (가끔 사용) 조금 더 공격적인 다운샘플링이 필요할 때 고려해볼 수 있습니다.
* `4x4`: (거의 사용 안 함) 정보 손실이 너무 커서 특별한 경우가 아니면 피하는 것이 좋습니다.

---
#### 필터의 갯수만큼 출력 층이 늘어난다

<img width="603" height="476" alt="image" src="https://github.com/user-attachments/assets/ccc46030-e93b-4e6c-8e3b-be446461301b" />

---


