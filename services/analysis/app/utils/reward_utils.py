import torch
import torch.nn.functional as F

def pairwise_loss(chosen_scores, rejected_scores):
    # Bradley–Terry style: log-sigmoid of (chosen - rejected)
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()

def kl_penalty(logprobs, ref_logprobs):
    return (logprobs - ref_logprobs).mean()
