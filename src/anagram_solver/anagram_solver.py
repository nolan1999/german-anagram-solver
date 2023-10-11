import copy
import numpy as np
import os
import sys
import torch
from src.model.model import CharRNN


CUDA = torch.cuda.is_available()
BEST_N = 50
N_RETRIES = 500


# Model
script_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_path, '..', 'model', 'model.net'), 'rb') as f:
    checkpoint = torch.load(f)
net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
net.load_state_dict(checkpoint['state_dict'])
if CUDA:
    net.cuda()
else:
    net.cpu()
net.eval()


def solve_anagram(chars):
    best_attempts = {}

    for char in chars:
        for _ in range(N_RETRIES):
            logprob = 0
            sentence = ""
            h = net.init_hidden(1)
            charset = copy.deepcopy(list(chars))
    
            while True:
                sentence += char
                if char != " ":
                    charset.remove(char)
                else:
                    if len([c for c in charset if c == " "]) > len(charset):
                        break  # avoid infinite loop
                if len(charset) == 0:
                    break
                probs, h = net.predict(char, h, cuda=CUDA)
                probs = {k: v for k, v in probs.items() if k in charset + [" "]}
                char = np.random.choice(
                    list(probs.keys()),
                    p=np.array(list(probs.values())) / sum(probs.values()),
                )
                logprob += np.log(probs[char])

            if len(best_attempts) < BEST_N:
                best_attempts[sentence] = logprob
            elif logprob > min(best_attempts.values()):
                worst_attempt = min(best_attempts, key=best_attempts.get)
                del best_attempts[worst_attempt]
                best_attempts[sentence] = logprob

    return {k: v for k, v in sorted(best_attempts.items(), key=lambda item: item[1], reverse=True)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m anagram_solver <chars>")
        exit(1)
    chars = sys.argv[1]
    for sentence, logprob in solve_anagram([c for c in chars]).items():
        print(f"{sentence} ({logprob})")
