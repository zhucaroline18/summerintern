import torch
a = torch.tensor([1.0,2.0,3.0], requires_grad=True)

b = a*a+3*a
loss = b.sum()
loss.backward()
print(a.grad)

optimizer.zero_grad()

loss.backward()
print(a.grad)