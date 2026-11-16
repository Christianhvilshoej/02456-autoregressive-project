import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from protgen.data.make_data import total_tokens

PAD, BOS, EOS = 0, 1, 2


@torch.no_grad()  # turn off gradient tracking as we are not training
def sample( model, num_sequences, min_len = 68, max_len=127, temperature=1.0, device="cpu"):
    model.eval()  # switch model to evaluation mode
    context_size = model.context_size # the number of previous tokens the model sees
    previous_tokens=torch.zeros(num_sequences, context_size, dtype=torch.long, device=device) # initialize placeholder for previous tokens
    previous_tokens[:, -1] = BOS 


    sequences = torch.zeros(num_sequences, max_len, dtype=torch.long, device=device) # initialize placeholder for generated sequences

    for i in range(max_len):
        logits = model(previous_tokens)/temperature # raw scores per tokens, per sequence
        logits[:, [PAD, BOS]] = float("-inf") # prevent the model from generating PAD and BOS tokens

        if i < min_len:
            logits[:, EOS] = float("-inf") # prevent generating EOS before minimum length is achieved
        next_token=Categorical(logits=logits).sample() # select new token based on logits
        sequences[:, i] = next_token 
        previous_tokens[:, :-1] = previous_tokens[:, 1:]
        previous_tokens[:, -1] = next_token
    return sequences

def sequences_to_strings(tokens):
    strings = []
    for row in tokens.cpu().tolist():
        if EOS in row:
            row = row[:row.index(EOS)] # drop everything from the first EOS on
        strings.append("".join(total_tokens[token] for token in row))
    return strings
