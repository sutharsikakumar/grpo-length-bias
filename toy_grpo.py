import torch
torch.manual_seed(0)

MAX_LEN = 10
GOOD, BAD, STOP = 0, 1, 2

def new_policy():
    return torch.zeros(MAX_LEN, 3, requires_grad=True)

policy = new_policy()
print(torch.softmax(policy[0], dim=-1))