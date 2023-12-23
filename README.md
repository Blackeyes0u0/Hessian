# Hessian


Autograd

```python
w1 = nn.Linear(3,3)
w2 = nn.Linear(3,3)
a = torch.tensor([[1.,2.,3.]])
o1 = w1(a)
o2 = w2(o1)
print(o1.grad_fn.next_functions)
print(o2.grad_fn.next_functions)

# AccmulatedGRad : bias
#  back ward할 길. # AddmmBackward
# TBackward0 : weight  
o2.grad_fn.next_functions[1][0] is o1.grad_fn 

# 보면 o2의 grad_next_functions는 o1.grad_fn에서 온것을 알수있다.
```


## 함수 모델

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 목표 함수
def target_function(x):
    return 2 * x + 3 + np.sin(x) * 3 * np.sqrt(abs(x)) + np.cos(x + 3)

# 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.linear(x)

# 모델 초기화
model = SimpleNet()

# 데이터 준비
x = torch.linspace(-10, 10, 1000).view(-1, 1)
y_true = torch.tensor(target_function(x.numpy()), dtype=torch.float32).view(-1, 1)

# 학습률
learning_rate = 0.01

# 학습 과정
for epoch in range(1000):
    # 순전파
    y_pred = model(x)

    # 손실 계산
    loss = nn.MSELoss()(y_pred, y_true)

    # 그래디언트 계산
    gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # 가중치 업데이트
    with torch.no_grad():
        for param, grad in zip(model.parameters(), gradients):
            param.data -= learning_rate * grad

    # 로깅
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 예측 및 그래프 그리기
x_train = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)
y_train = torch.tensor(target_function(x_train.numpy()), dtype=torch.float32).view(-1, 1)
predicted = model(x_train).detach().numpy()

# 목표 함수와 학습 결과 비교를 위한 그래프
plt.figure(figsize=(10,6))
plt.plot(x_train.numpy(), y_train.numpy(), label='Target Function', color='blue')
plt.plot(x_train.numpy(), predicted, label='Predicted', color='red')
```

```hash
# output
Epoch 0, Loss: 198.77162170410156
Epoch 200, Loss: 38.84187316894531
Epoch 400, Loss: 14.687801361083984
Epoch 600, Loss: 15.193655967712402
Epoch 800, Loss: 8.485503196716309
```


## Retain Graph (DAG)

```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

## High Order Derivatives

```python
import torch

# 임의의 함수 f(x) = x^3
def f(x):
    return x ** 4 #+torch.sin(x) +torch.cos(x) + torch.exp(x)

# 임의의 입력값 x
x = torch.tensor(1.0, requires_grad=True)

# f(x)에 대한 첫 번째 미분 계산
first_derivative = torch.autograd.grad(f(x), x, create_graph=True)[0]

# create_graph=True를 사용하면 미분에 대한 미분을 계산할 수 있습니다.
# 즉, higher order derivatives을 계산할 수 있습니다.
print(f"First derivative at x={x}: {first_derivative}")

# 첫 번째 미분에 대한 또 다른 미분(즉, 2차 미분) 계산
second_derivative = torch.autograd.grad(first_derivative, x,create_graph=True)[0]
print(f"Second derivative at x={x}: {second_derivative}")

# third derivative
third_derivative = torch.autograd.grad(second_derivative, x)[0]
print(f"Third derivative at x={x}: {third_derivative}")
```

# Hessian Matrix 

## 1. Hessian 벡터 곱(Hessian-vector product)

```python
import torch

def compute_hvp(loss, parameters, vector): # parameters : List(torch.Tensor)
    # 첫 번째 그래디언트 계산
    grads = torch.autograd.grad(loss, parameters, create_graph=True)
    
    # grads와 vector의 내적 계산
    grad_vector_product = sum(torch.sum(g * v) for g, v in zip(grads, vector))

    # 내적에 대한 그래디언트 계산 (Hessian 벡터 곱)
    hvp = torch.autograd.grad(grad_vector_product, parameters)
    
    return [v.detach() for v in hvp]

# 예시: 모델과 데이터
def target_function(x):
    return 2 * x + 3 + np.sin(x) * 3 * np.sqrt(abs(x)) + np.cos(x + 3)

# 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.linear(x)

# 모델 초기화
model = SimpleNet()

input_data = torch.linspace(-10, 10, 1000).view(-1, 1)
target =  torch.tensor(target_function(input_data.numpy()), dtype=torch.float32).view(-1, 1)

# 손실 계산
output = model(input_data)
loss = torch.nn.MSELoss()(output, target)


params = list(model.parameters())

vector = [torch.randn_like(p) for p in params]  # 랜덤 벡터
hvp = compute_hvp(loss, params, vector)

for x,y,z in zip(params,vector,hvp):
    print(x.shape,y.shape,z.shape)
print(x,y,z)
```

## 2. Hessian Matrix Diagonal term

```python
import torch

# 예시: 모델과 데이터
def target_function(x):
    return 2 * x + 3 + np.sin(x) * 3 * np.sqrt(abs(x)) + np.cos(x + 3)

# 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.linear(x)

# 모델 초기화
model = SimpleNet()

input_data = torch.linspace(-10, 10, 1000).view(-1, 1)
target =  torch.tensor(target_function(input_data.numpy()), dtype=torch.float32).view(-1, 1)

# 손실 계산
output = model(input_data)
loss = torch.nn.MSELoss()(output, target)

params = {}
jacobian = {}
hessian_digonal = {}

    # please differentiate the jacobian
    # jaco shoud be a scalar tensor
    # fill the hessian_digonal


for name, param in model.named_parameters():
    params[name] = param
    jaco = torch.autograd.grad(loss, param, create_graph=True)[0]
    jacobian[name] = jaco

    # Hessian Diagonal term
    # Hessian 대각 성분 계산
    hessian_diag = []
    for j in jaco.view(-1):
        # jaco는 텐서이므로, 각 요소에 대해 두 번째 미분을 계산
        hess = torch.autograd.grad(j, param, retain_graph=True)[0]
        hessian_diag.append(hess.view(-1))

    # 대각 성분을 하나의 벡터로 결합
    hessian_digonal[name] = torch.cat(hessian_diag)

hessian_digonal
```

