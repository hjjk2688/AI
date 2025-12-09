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

<img width="532" height="275" alt="image" src="https://github.com/user-attachments/assets/1529e1a4-991f-4d49-80b0-a5e0126bd39d" />
<img width="625" height="212" alt="image" src="https://github.com/user-attachments/assets/b45b3725-a741-4798-b6f5-d0389c6b45cd" />

<img width="481" height="137" alt="image" src="https://github.com/user-attachments/assets/c163e4d1-a2e9-473a-bc16-4a6e60659c8e" />

- w : 가중치 , t : 역치(낮으면 쉽게 넘어가고 높으면 넘어가기어렵다.)

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

<img width="1231" height="246" alt="image" src="https://github.com/user-attachments/assets/5a8d388b-9fae-4361-9e7e-24fcc1ef994b" />

<img width="825" height="270" alt="image" src="https://github.com/user-attachments/assets/013cad7d-29ec-4218-ae04-1f96f1196c31" />

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
