# AI Basic

## 인공 신공망 구조

1. 심층 신명망(DNH)

<img width="921" height="377" alt="image" src="https://github.com/user-attachments/assets/58430250-44d8-41af-aad7-6ff860ffb3eb" />

2. 합성곱 신경망 (CNN)

<img width="956" height="405" alt="image" src="https://github.com/user-attachments/assets/97a9c798-552e-4153-b432-413471f6e7a1" />

3. 반복 신경 망(RNN) 

<img width="643" height="414" alt="image" src="https://github.com/user-attachments/assets/520ee241-57b2-48c6-a34f-2c6c1c67f1db" />

## 가장 간단한 인공 신경

<img width="813" height="408" alt="image" src="https://github.com/user-attachments/assets/149fd44a-453e-4360-ae9f-2ac5e9593788" />

- 가지동기 : 가중치
- 촉살돌기 : 편향 (민감도)

<img width="656" height="437" alt="image" src="https://github.com/user-attachments/assets/0e28ed83-003c-465a-8283-9e2e0c7c95f7" />


<img width="466" height="248" alt="image" src="https://github.com/user-attachments/assets/35e4b538-ca00-4ba1-a7ce-42b0f2c78817" />

```text
y = x*w + 1*b

w : 가중치 
b : 편향 (민감도)
```

### 학습 

- 들어가는 데이터(값) 나오는 데이터(값)은 정해져 있다.
- w , b 를 이용해서 x 를 y 로 만드는 과정

### 활성화 함수

<img width="587" height="385" alt="image" src="https://github.com/user-attachments/assets/d0245df4-4e81-467e-9349-1ea74a1cb5c5" />

1. sigmoid 함수 : 출력 값 0 ~ 1
2. relu 함수 : 0 보다 큰 출력값

## 퍼셉트론의 이해

<table>
    <tr>
        <td><img width="532" height="275" alt="image" src="https://github.com/user-attachments/assets/1529e1a4-991f-4d49-80b0-a5e0126bd39d" /></td>
        <td><img width="625" height="212" alt="image" src="https://github.com/user-attachments/assets/b45b3725-a741-4798-b6f5-d0389c6b45cd" /></td>
        <td><img width="481" height="137" alt="image" src="https://github.com/user-attachments/assets/c163e4d1-a2e9-473a-bc16-4a6e60659c8e" /></td>
    </tr>
</table>

- w : 가중치 , t : 역치(낮으면 쉽게 넘어가고 높으면 넘어가기어렵다.)
```
가중치 w (Weight)의 역할: 영향력 조절 (기울기)

- 가중치 w는 입력 `x`가 출력 `y`에 얼마나 큰 영향을 미칠지를 결정합니다. 즉, 직선의 기울기 역할을 합니다.
- w의 절댓값이 크면, 입력 x가 조금만 변해도 출력 y가 크게 변합니다. (입력의 영향력이 크다)
- w의 절댓값이 작으면, 입력 x가 많이 변해도 출력 y는 조금만 변합니다. (입력의 영향력이 작다)

이처럼 w는 입력 `x`와 직접 상호작용하여 그 중요도나 영향력을 조절하기 때문에, x와 곱해지는 형태로 사용됩니다.
```
```
편향 b (Bias)의 역할: 기본 활성도 조절 (y절편)

- 편향 b는 입력 `x`가 0일 때의 기본 출력값을 결정합니다. 즉, 직선의 y절편 역할을 합니다.
- b는 입력 x의 값과 상관없이, 계산 결과 전체를 위아래로 평행 이동시키는 역할을 합니다.
- 이는 뉴런이 얼마나 쉽게 활성화될지를 결정합니다. b가 크면 작은 입력에도 뉴런이 쉽게 활성화되고,
  b가 매우작으면 큰 입력이 들어와야만 활성화됩니다.

이처럼 b는 입력과 독립적으로 뉴런의 기본 활성도를 조절하기 때문에, 그냥 더해지는 형태로 사용됩니다.
```
#### 편향값에 곱해준는 상수가 항상 1인 이유
```
y = w*x + b 라는 식에서 b는 입력(`x`)이 얼마이든 상관없이 결과에 더해지는 값입니다. 이러한 역할을
수학적으로 가장 깔끔하게 표현하는 방법이 바로 "크기가 1인 가상의 입력이 항상 존재하고, 그 입력에 대한
가중치가 b이다"라고 생각하는 것입니다.
```
### 퍼셉트론의 한계 : XOR

<img width="1240" height="355" alt="image" src="https://github.com/user-attachments/assets/2c6b36ab-dad2-41d5-8964-9087dc91e652" />

```Text
(1)식은 (2) 식과 같이 되어 b는 0보다 작거나 같습니다. (3), (4) 식을 더하면 (5) 식과 같게 됩니다.
(6) 식은 양변에 마이너스(-)를 붙이며 (7) 식과 같아집니다.
(5) 식과 (7) 식을 더하면 w1, w2는 상쇄되어 없어지고 (8) 식과 같이 되어 b는 0보다 큽니다.
결과적으로 (2) 식과 (8) 식은 동시에 만족할 수 없습니다.
```

### 다층 퍼셉트론으로 해결 : XOR

- XOR 게이트를 NAND, OR, AND 게이트를 조합하여 해결

<table>
    <tr>
        <td><img width="1231" height="246" alt="image" src="https://github.com/user-attachments/assets/5a8d388b-9fae-4361-9e7e-24fcc1ef994b" /></td>
        <td><img width="825" height="270" alt="image" src="https://github.com/user-attachments/assets/013cad7d-29ec-4218-ae04-1f96f1196c31" /></td>
    </tr>
</table>

```text
다층 퍼셉트론은 입력층, 은닉층, 출력층으로 구성된 모델입니다.
XOR 게이트의 경우 두 개 의 입력값과 하나의 출력값을 가진 단층 퍼셉트론 세 개를 조합하여
다층 퍼셉트론을 만들어 해결할 수 있습니다.
```

```python
def AND(x1,x2):
    w1,w2 = 0.4,0.4;
    b = -0.6;
    s = (x1 * w1) + (x2 * w2) + b;
    return 0 if s <= 0 else 1;

def NAND(x1, x2):
    w1, w2 = -0.5, -0.5;
    b = 1;
    s = (x1 * w1) + (x2 * w2) + b;
    return 0 if s <= 0 else 1;

def OR(x1, x2):
    w1 , w2 = 0.6, 0.6;
    b = -0.5;
    s = (x1 * w1) + (x2 * w2) + b;
    return 0 if s <= 0 else 1;

def XOR(x1, x2):
    h1 = OR(x1,x2);
    h2 = NAND(x1, x2);
    return AND(h1, h2);

print(XOR(0,0), XOR(0,1), XOR(1,0), XOR(1,1));
```

<img width="415" height="133" alt="image" src="https://github.com/user-attachments/assets/0b03618d-af37-4067-aaa9-ba55c7f83fc5" />

# 딥러닝 7 공식

<img width="1185" height="395" alt="image" src="https://github.com/user-attachments/assets/e3484f89-1602-4e92-bd66-1b3aa5ac0a31" />

- 순전파 , 역전파

## 제 1 공식 : 순전파

<img width="1130" height="269" alt="image" src="https://github.com/user-attachments/assets/a9bb5924-02c4-45a4-9901-129bf6e366b3" />

- ex) x = 2 , w = 3 , b = 1  => y = 7 

<img width="436" height="195" alt="image" src="https://github.com/user-attachments/assets/b3d416d2-d0eb-47f5-8a5b-65e01776071f" />

## 제 2 공식 : 평균 제곱 오차 

<img width="459" height="61" alt="image" src="https://github.com/user-attachments/assets/39013ac9-1042-4ad5-8990-3d260720a157" />

- E(오차) 를 0으로 만들어가야 한다. ( y = yT)
- E : 오차, y = 순전파에 의한 예측값, yT: 목표값(라벨)
- yT는 입력값 x에 대해 실제로 나오기를 원하는 값

- y값으로 7을 얻었을때 y로 10이 나오게 하고 싶습니다. 이 경우 yT 값은 10이 됩니다. 그러면 평균 제곱 오차는 다음과 같이 계산됩니다.
```
E = (7 – 10)*(7 – 10)/2 = (-3)*(-3)/2 = 9/2 = 4.5
```

## 제 3 공식 : 역전파 오차

- E에 대한 y의 기울기(미분)

<img width="265" height="48" alt="image" src="https://github.com/user-attachments/assets/1932e757-0d04-4471-8a75-a2d1a74255e9" />

- yE(델다E/델타y) : 역전파 오차, y : 순전파에 의한 예측값, yT : 목표값(라벨)
- yE의 정확한 의미는 y에 대한 오차 E의 순간변화율을 의미하며 편미분을 통해 유도

#### y = 7 , y 가 10이 나오게 하고싶을때
- y값이 10이 되려면 3이 모자랍니다. y의 오차는 w와 b의 오차로 인해 발생합니다. 따라서 w와 b값을 적당히 증가시키면 y로 10에 가까운 값이 나오게 할 수 있다.

<table>
    <tr>
        <td><img width="629" height="285" alt="image" src="https://github.com/user-attachments/assets/8560464e-2520-4d24-ab6c-47b92170e35f" /></td>
        <td><img width="697" height="320" alt="image" src="https://github.com/user-attachments/assets/6f5a6dce-8165-4e04-aaed-ebacac2fe784" /></td>
    </tr>
</table>

## 제 4 공식 : 입력 역전파

<img width="1034" height="297" alt="image" src="https://github.com/user-attachments/assets/59591e8a-97be-4514-b95d-2793b8c1c7e1" />

- xE는 입력 역전파, yE는 역전파 오차로 딥러닝 제 3 공식에서 구한 값입니다. 회색으로 표시된 1E는 숫자 1의 오차라는 의미로 사용하지 않는 부분입니다. 
  딥러닝 제 4 공식은 다음과 같은 순서로 유도할 수 있습니다.

1. 딥러닝 제 1 공식의 그림을 복사합니다.
2. y -> yE, x -> xE로 변경합니다.
3. 화살표 방향을 반대로 합니다.
4. 1E, b는 사용하지 않습니다.

<img width="822" height="299" alt="image" src="https://github.com/user-attachments/assets/d9fada67-dce5-4aa6-8660-9f1dba2200a4" />

```
순전파에서 x는 w를 따라서 y로 전파됩니다. y로 전파된 예측 값은 목표 값과 비교하는 과정에서 오차(yE)가 발생합니다.
발생한 오차는 궁극적으로 0이 되도록 만들어야 합니다. 그러기 위해서 오차를 어딘가로 보내야 하는데, 어디로 보내야 할까요?
답은 순전파에서 통로로 사용된 w를 따라서 거꾸로 전파(xE)합니다.
```

## 제 5 공식 : 가중치, 편향 순전파

<img width="974" height="328" alt="image" src="https://github.com/user-attachments/assets/c5be0b8c-574d-4deb-ae92-b83668fe6761" />

- 딥러닝 제 5공식 유도

1. 딥러닝 제 1 공식을 복사합니다.
2. x와 w, 1과 b를 교환하여 딥러닝 제 5 공식을 유도합니다.
3. 딥러닝 제 1 공식의 그림과 같은 형태로 그림을 그립니다.

<img width="822" height="243" alt="image" src="https://github.com/user-attachments/assets/8ef55772-545c-4965-9619-1d9ba30973c4" />

## 제 6 공식 : 가중치, 편향 역전파

<img width="801" height="256" alt="image" src="https://github.com/user-attachments/assets/4241bf25-0b32-4866-b6bf-fe19793591b3" />

- wE : 가중치 역전파 오차, bE : 편향 역전파 오차, yE : 역전파 오차(제 3공식에서 구한값)

- 딥러닝 제 6 공식 유도

1. 딥러닝 제 5 공식의 그림을 복사합니다.
2. y -> yE, w -> wE, b –> bE로 변경합니다.
3. 화살표 방향을 반대로 합니다.

<img width="843" height="270" alt="image" src="https://github.com/user-attachments/assets/fe9cd7d7-2af0-4505-bd16-8c2869ad13a8" />

## 제 7 공식 : 신경망 학습 

<img width="317" height="118" alt="image" src="https://github.com/user-attachments/assets/03e02827-55ed-442a-aad7-b34e205665e4" />

- lr : 학습률(learning rate), wE: 가중치 역전파 오차, bE: 편향 역전파 오차
- 제 7 공식은 경사하강법, 미분을 이용하여 얻은 공식

#### 학습률 정하는법

- 학습률의 역할은 '경사 하강법'이라는 언덕 내려가기 게임에서 보폭(step size)을 얼마나 크게 할지 결정하는것과 같습니다.
    - 학습률이 너무 크면: 보폭이 너무 커서 언덕의 가장 낮은 지점(최적점)을 휙 지나쳐 버리거나, 오히려 반대편으로 넘어가 발산해버릴 수 있습니다. (학습 실패)
    - 학습률이 너무 작으면: 보폭이 너무 작아서 언덕을 내려가는 데 시간이 매우 오래 걸리거나,
      넓고 평평한 최적점에 도달하기 전에 좁고 움푹 팬 곳(지역 최적점, local minimum)에 갇혀버릴 수 있습니다.

>일반적으로는 `0.01` 또는 `0.001` 로 시작해서 결과를 보고 조절하는 경우가 가장 많습니다 (보통 adam 사용)
>경사 하강법에서는 손실 함수(Loss Function)의 기울기가 0이 되는 지점(최적점)을 찾아가는 것이 목표
- ex) 기울기가 완만할수록 변화가 적고 0에 가깝다, 기울기가 가파른수록 변화가크고 1에 가깝다.

```python
x = 2
w = 3
b = 1
yT = 10
lr = 0.01

for epoch in range(250):
    
    y = (x*w) + (1*b)
    E = (y - yT)**2 / 2
    yE = y - yT
    wE = yE*x
    bE = yE*1
    w -= lr*wE
    b -= lr*bE
    
    print(f'epoch = {epoch}')
    print(f' y : {y:.3f}')
    print(f' w : {w:.3f}')
    print(f' b : {b:.3f}')

```

- epoch = 200일때 목표 10 달성

<img width="759" height="172" alt="image" src="https://github.com/user-attachments/assets/7e261b17-e732-4286-bc91-9f84b2de42ae" />

<img width="833" height="193" alt="image" src="https://github.com/user-attachments/assets/1e652bb2-e5ff-4cf8-a746-ab38a11a1770" />

- lr = 0.05 변경 epoch = 27에 목표에 거의 근사

<img width="774" height="161" alt="image" src="https://github.com/user-attachments/assets/ba9f7779-68fa-457f-bf3d-70de222020fa" />

#### 오차 E가 충분히 작아 지면 학습 중단 코드 추가
```python
x = 2
w = 3
b = 1
yT = 10
lr = 0.01

for epoch in range(250):
    
    y = (x*w) + (1*b)
    E = (y - yT)**2 / 2
    yE = y - yT
    wE = yE*x
    bE = yE*1
    w -= lr*wE
    b -= lr*bE
    
    print(f'epoch = {epoch}')
    print(f' y : {y:.3f}')
    print(f' w : {w:.3f}')
    print(f' b : {b:.3f}')

    if E < 0.0000001 :
    break
```

## 딥러닝 7 공식 정리
<img width="1233" height="293" alt="image" src="https://github.com/user-attachments/assets/88e1b9dc-4a2d-470b-9eb9-9847d8d2e5bb" />

wE = 델타e/델타w (기울기)

#### 입력 3개 , 목표 27

<img width="1245" height="378" alt="image" src="https://github.com/user-attachments/assets/4b3f08c9-beb5-4875-8553-448907de20a1" />

<img width="562" height="294" alt="image" src="https://github.com/user-attachments/assets/57848a8a-34a2-4824-b03b-69f4b8a998ae" />

```python
x1, x2 = 2, 3
w1, w2 = 3, 4
b = 1
yT = 27
lr = 0.01

for epoch in range(200):
    
    y = (x1*w1) + (x2*w2) + (1*b)
    E = (y - yT)**2 / 2
    yE = y - yT
    w1E = yE*x1
    w2E = yE*x2
    bE = yE*1
    w1 -= lr*w1E
    w2 -= lr*w2E
    b -= lr*bE
    
    print(f'epoch = {epoch}')
    print(f' y : {y:.3f}')
    print(f' w1 : {w1:.3f}')
    print(f' w2 : {w2:.3f}')
    print(f' b : {b:.3f}')
    
    if E < 0.0000001: # 오차가 백만분의1이 되면 학습을 멈춘다.
        break;

```
- epoch 65일때 목표치 근사값 달성

<img width="772" height="170" alt="image" src="https://github.com/user-attachments/assets/b7618df2-c9f2-4620-93e4-636de91d7565" />

#### 입력2 출력2

<img width="1106" height="330" alt="image" src="https://github.com/user-attachments/assets/55488bba-eb7c-498e-9cb7-0206de44fb6a" />

```python
x1, x2 = 2, 3
w1, w2 = 3, 4
w3, w4 = 5, 6
b1, b2 = 1, 2
y1T, y2T = 27, -30
lr = 0.01

for epoch in range(200):
    
    y1 = (x1*w1) + (x2*w2) + (1*b1)
    y2 = (x1*w3) + (x2*w4) + (2*b2)
    E = (y1 - y1T)**2 /2 + (y2 - y2T)**2 / 2
    y1E = y1 - y1T
    y2E = y2 - y2T
    w1E = y1E*x1
    w2E = y1E*x2
    b1E = y1E*1
    w3E = y2E*x1
    w4E = y2E*x2
    b2E = y2E*1
    w1 -= lr*w1E
    w2 -= lr*w2E
    b1 -= lr*b1E
    w3 -= lr*w3E
    w4 -= lr*w4E
    b2 -= lr*b2E
    
    print(f'epoch = {epoch}')
    print(f' y1 : {y1:.3f}')
    print(f' y2 : {y2:.3f}')
    print(f' w1 : {w1:.3f}')
    print(f' w2 : {w2:.3f}')
    print(f' b1 : {b1:.3f}')
    print(f' w3 : {w3:.3f}')
    print(f' w4 : {w4:.3f}')
    print(f' b2 : {b2:.3f}')
    
    if E < 0.0000001:
        break;

```

<img width="936" height="382" alt="image" src="https://github.com/user-attachments/assets/897c3683-4337-4048-a183-570b2c4d6055" />


### 입력2 출력3

<img width="896" height="513" alt="image" src="https://github.com/user-attachments/assets/7de92a47-5758-4364-b79b-fddfc0107f3c" />

- lr = 0.01

```python
x1, x2 = 0.05, 0.10
w1, w2 = 0.15, 0.20
w3, w4 = 0.25, 0.30
w5, w6 = 0.40, 0.55
b1, b2, b3 = 0.35, 0.45, 0.60
y1T, y2T, y3T = 0.01, 0.99, 0.50
lr = 0.01

for epoch in range(1000):
    
    y1 = (x1*w1) + (x2*w2) + (1*b1)
    y2 = (x1*w3) + (x2*w4) + (1*b2)
    y3 = (x1*w5) + (x2*w6) + (1*b3)
    #E = (((y1 - y1T)**2)/2) + (((y2 - y2T)**2)/2) + (((y3-y3T)**2) /2) #미분 평의성
    E = (((y1 - y1T)**2) + ((y2 - y2T)**2) + ((y3-y3T)**2)) / 3 # 평균제곱 오차 
    y1E = y1 - y1T
    y2E = y2 - y2T
    y3E = y3 - y3T
    
    w1E = y1E*x1
    w2E = y1E*x2
    b1E = y1E*1
    
    w3E = y2E*x1
    w4E = y2E*x2
    b2E = y2E*1
    
    w5E = y3E*x1
    w6E = y3E*x2
    b3E = y3E*1
    
    w1 -= lr*w1E
    w2 -= lr*w2E
    b1 -= lr*b1E
    
    w3 -= lr*w3E
    w4 -= lr*w4E
    b2 -= lr*b2E
    
    w5 -= lr*w5E
    w6 -= lr*w6E
    b3 -= lr*b3E
    
    print(f'epoch = {epoch}')
    print(f' y1 : {y1:.3f}')
    print(f' y2 : {y2:.3f}')
    print(f' y3 : {y3:.3f}')
    print(f'---------------')
    print(f' w1 : {w1:.3f}')
    print(f' w2 : {w2:.3f}')
    print(f' b1 : {b1:.3f}')
    print(f'---------------')
    print(f' w3 : {w3:.3f}')
    print(f' w4 : {w4:.3f}')
    print(f' b2 : {b2:.3f}')
    print(f'---------------')
    print(f' w5 : {w6:.3f}')
    print(f' w6 : {w5:.3f}')
    print(f' b3 : {b3:.3f}')
    
    if E < 0.0000001:
        break;

```
<img width="779" height="346" alt="image" src="https://github.com/user-attachments/assets/75cd86c3-b306-4aa0-9afd-f30bcf6697c5" />

#### 3입력 2출력

<img width="1014" height="643" alt="image" src="https://github.com/user-attachments/assets/abaec2b7-07e3-47fc-9c9d-91c9d28c88ea" />

- lr = 0.01

<img width="517" height="285" alt="image" src="https://github.com/user-attachments/assets/95cdd10f-5ff9-439d-a7ff-2b9de8e41f9c" />

---
## 평균제곱 오차 vs 미분편의성 (제2공식)
ex) 입력2 출력3
```
1번 E = (((y1 - y1T)**2)/2) + (((y2 - y2T)**2)/2) + (((y3-y3T)**2) /2)
2번 E = (((y1 - y1T)**2) + ((y2 - y2T)**2) + ((y3-y3T)**2)) / 3
```
1번 - 미분 편의성, 2번 - 평균제곱 오차


## 손실 함수: 평균 제곱 오차(MSE)와 1/2 제곱합 오차 비교

두 손실 함수는 모두 예측값과 실제값의 차이를 제곱하여 오차를 계산하는 **제곱 오차(Squared Error)**에 기반하지만, 오차의 총합을 어떤 상수로 나누는지에 따라 그 목적과 해석이 달라집니다.

### 핵심 비교

| 구분 | 평균 제곱 오차 (MSE) | 1/2 제곱합 오차 |
| :--- | :--- | :--- |
| **수식** | `E = (1/n) * Σ(y - y_T)²` | `E = (1/2) * Σ(y - y_T)²` |
| **핵심 목적** | 오차의 통계적 **'평균'** 계산 | 역전파 시 **'미분 계산의 편의성'** |
| **나누는 값 `n` 또는 `2`의 의미** | `n` = **출력의 개수**<br>(데이터 개수에 따른 평균) | `2` = **고정된 상수**<br>(미분 공식을 위한 수학적 트릭) |
| **주요 사용처** | • 모델의 성능을 통계적으로 해석<br>• 다른 모델과 성능을 비교<br>• PyTorch, TensorFlow 등 라이브러리의 기본 손실 함수 | • 경사 하강법의 원리를 설명하는 교과서, 논문, 강의<br>• 기울기 계산을 직접 구현할 때 |

---

### 왜 1/2을 곱하는가? (미분과의 관계)

이것이 '미분 편의성'의 핵심 이유입니다.

1.  **기본 제곱 오차항**
    하나의 출력에 대한 제곱 오차는 `E = (y - y_T)²` 입니다.

2.  **기본 오차항의 미분**
    역전파 시, 이 오차를 예측값 `y`에 대해 미분하여 기울기(gradient)를 구해야 합니다. 연쇄 법칙(chain rule)에 따라 미분하면 다음과 같습니다.
    
    `dE/dy = 2 * (y - y_T)¹ * (y의 미분)`
    
    `dE/dy = 2 * (y - y_T)`
    
    결과적으로 기울기 앞에 숫자 `2`가 붙게 됩니다.

3.  **1/2을 적용한 오차항의 미분**
    이제 오차 함수를 `E = (1/2) * (y - y_T)²` 라고 정의해 보겠습니다. 이것을 `y`에 대해 미분하면,
    
    `dE/dy = (1/2) * 2 * (y - y_T)`
    
    `dE/dy = y - y_T`
    
    앞에 곱해진 `1/2`과 미분해서 나온 `2`가 서로 약분되어 사라지면서, 기울기가 `y - y_T` 라는 매우 깔끔한 형태로 남게 됩니다. 이처럼 공식을 단순화하여 이론을 설명하고 계산을 편리하게 만들기 위해 `1/2`을 곱해주는 것입니다.

---

### 언제 무엇을 사용해야 할까?

*   **평균 제곱 오차 (MSE)**: 모델의 성능을 최종적으로 보고하거나, 다른 모델과 성능을 객관적으로 비교하고 싶을 때 사용합니다. 출력 개수가 다른 모델이라도 '평균' 오차를 비교할 수 있기 때문입니다. **실제 코드 구현에서는 라이브러리의 기본값인 MSE를 사용하는 것이 가장 일반적입니다.**
    *    평균 제곱 오차는 보통 회귀(Regression) 문제에 사용됩니다. 회귀란, 연속적인 숫자 값을 예측하는 문제입니다. (예: 집값 예측, 주가 예측)

*   **1/2 제곱합 오차**: 주로 이론적인 배경을 설명하거나 학습할 때 등장합니다. 실제 최적화 과정에서는 어떤 상수를 곱하든 학습률(learning rate)이 그 차이를 흡수하므로 최종 결과에는 큰 영향을 미치지 않습니다. 하지만 '왜 2로 나누는가?'에 대한 질문에는 '미분을 깔끔하게 만들기 위해서'라고 이해하는 것이 핵심입니다.

---

## 2입력 2 은닉 2출력 인공 신공망

<table>
    <tr>
        <td><img width="801" height="433" alt="image" src="https://github.com/user-attachments/assets/27a1117d-d08b-4283-a4e5-6668045bb6bd" /></td>        
        <td><img width="806" height="339" alt="image" src="https://github.com/user-attachments/assets/0ca8e9ac-ed9c-4beb-900c-a9eca2189413" /></td>
    </tr>   
</table>

```python
x1, x2 = 0.05, 0.10
w1, w2 = 0.15, 0.20
w3, w4 = 0.25, 0.30
w5, w6 = 0.40, 0.45
w7, w8 = 0.50, 0.55
b1, b2, b3, b4 = 0.35, 0.35, 0.60, 0.60
y1T, y2T= 0.01, 0.99
lr = 0.01

for epoch in range(1000):
    
    h1 = (x1*w1) + (x2*w2) + (1*b1)
    h2 = (x1*w3) + (x2*w4) + (1*b2)
    
    y1 = (h1*w5) + (h2*w6) + (1*b3)
    y2 = (h1*w7) + (h2*w8) + (1*b4)
    
    E = ((y1 - y1T)**2) / 2  + ((y2 - y2T)**2) / 2
    
    y1E = y1 - y1T
    y2E = y2 - y2T
  
    h1E = (y1E*w5) + (y2E*w7) 
    h2E = (y1E*w6) + (y1E*w8)
    
    w5E = y1E*h1
    w6E = y1E*h2
    w7E = y2E*h1
    w8E = y2E*h2
    b3E = y1E*1
    b4E = y2E*1
    
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
    

    print(f'epoch = {epoch}')
    print(f' y1 : {y1:.3f}')
    print(f' y2 : {y2:.3f}')
    
    if E < 0.0000001:
        break;

```

<img width="501" height="152" alt="image" src="https://github.com/user-attachments/assets/a6ab7830-f56b-40cb-bbae-505f4d00cbba" />

