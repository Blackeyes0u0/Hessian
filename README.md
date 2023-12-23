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
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 목표 함수 정의
def target_function(x):
    return 2 * x + 3 + np.sin(x)*3*np.sqrt(abs(x)) +np.cos(x+3)

# 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(10,10),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(10,1),
        )
    def forward(self, x):
        return self.linear(x)

# 데이터 생성
x_train = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)
y_train = target_function(x_train)

# 모델, 손실 함수, 옵티마이저 초기화
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 루프
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % (num_epochs//5) == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 학습된 모델로 예측
predicted = model(x_train).detach().numpy()

# 목표 함수와 학습 결과 비교를 위한 그래프
plt.figure(figsize=(10,6))
plt.plot(x_train.numpy(), y_train.numpy(), label='Target Function', color='blue')
plt.plot(x_train.numpy(), predicted, label='Model Predictions', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Target Function and Model Predictions')
plt.show()
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

