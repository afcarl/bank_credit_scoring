import torch

loss_fn = torch.nn.MSELoss()
x = torch.autograd.Variable(torch.zeros(1, 5), requires_grad=True)
mean = x.mean(-1, keepdim=True)
mean.retain_grad()
std = x.std(-1, keepdim=True)
std.retain_grad()
r = (x - mean) / (std + 1e-6)
r.retain_grad()
loss = loss_fn(r, torch.autograd.Variable(torch.ones(1, 5)))
loss.retain_grad()
loss.backward()
print(loss.grad)