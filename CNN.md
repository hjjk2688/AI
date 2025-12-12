# CNN

## CNN(Convolution Neural Network)

<img width="836" height="395" alt="image" src="https://github.com/user-attachments/assets/98cba750-17a6-4f00-9b13-49b881ad8b24" />

- kernel = filter(직육면체 가중치)
- 기본 데이터 형태: 가로x세로x깊이 
- 각 특징을 추출하는 filter(convolution)는 특징의 수만큼 있어야된다. ( 각 특징을 추출(곱해서) 점을 찍어 새로운 이미지를 생성)
- Pooling: 그림의 크기를 1/4로 줄이는작업(연산을 줄이기위해서, 가장 큰 특징을 선택)


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
