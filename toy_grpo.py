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

def reward(tokens):
    return 1.0 if BAD not in tokens else 0.0

G = 8 

def grpo_step(policy, optimizer, variant):
    samples = [sample_response(policy) for _ in range(G)]
    rewards = torch.tensor([reward(tok) for tok, _ in samples])

    if variant == "A":   # original GRPO
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        losses = [-(a * lp).mean() for a, (_, lp) in zip(adv, samples)] 
        loss = torch.stack(losses).mean()
    else:
        adv = rewards - rewards.mean()       
        losses = [-(a * lp).sum() for a, (_, lp) in zip(adv, samples)] 
        loss = torch.stack(losses).sum() / (G * (MAX_LEN + 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return samples, rewards

policy = new_policy()
opt = torch.optim.Adam([policy], lr=0.05)
samples, rewards = grpo_step(policy, opt, "A")
print(rewards)
print(torch.softmax(policy[0], dim=-1)) 

def reward(tokens):
    return 1.0 if len(tokens) > 0 and BAD not in tokens else 0.0