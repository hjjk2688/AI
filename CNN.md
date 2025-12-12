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
3x3 입력: filter size

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



