import torch
torch.manual_seed(0)

MAX_LEN = 10
GOOD, BAD, STOP = 0, 1, 2

def new_policy():
    return torch.zeros(MAX_LEN, 3, requires_grad=True)

policy = new_policy()
print(torch.softmax(policy[0], dim=-1))

def sample_response(policy):
    tokens, logprobs = [], []
    for t in range(MAX_LEN):
        probs = torch.softmax(policy[t], dim=-1)     
        dist = torch.distributions.Categorical(probs)    
        action = dist.sample()                      
        logprobs.append(dist.log_prob(action))  
        if action.item() == STOP:
            break
        tokens.append(action.item())
    return tokens, torch.stack(logprobs)

policy = new_policy()
for _ in range(5):
    tokens, lp = sample_response(policy)
    print(tokens, lp.sum().item())