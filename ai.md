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

