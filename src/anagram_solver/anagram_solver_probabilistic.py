import copy
import json
import numpy as np
import os
import sys
import torch
from src.model.model import CharRNN


CUDA = torch.cuda.is_available()
BEST_N = 50
N_RETRIES = 500


def _get_model():
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'model.net')
    with open(ckpt_path, 'rb') as f:
        checkpoint = torch.load(f)
    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])
    if CUDA:
        net.cuda()
    else:
        net.cpu()
    net.eval()
    return net


def _get_word_check(check=True):
    words_path = os.path.join(os.path.dirname(__file__), "..", "data", "words.txt")
    with open(words_path, "r") as f:
        words = set([w.strip().lower() for w in f.readlines()])
    is_word = lambda w: w.lower() in words if check else True
    return is_word


def _get_letter_priors():
    priors_path = os.path.join(os.path.dirname(__file__), "..", "data", "letter_priors.json")
    with open(priors_path, "r") as f:
        letter_priors = json.load(f)
    return letter_priors


def solve_anagram(chars, check_words=True):
    is_word = _get_word_check(check=check_words)
    letter_priors = _get_letter_priors()
    net = _get_model()

    best_attempt = None

    while True:
        sentence = ""
        last_word = ""
        charset = copy.deepcopy(list(chars))
        char = "\n"
        h = net.init_hidden(1)
        logprob = 0

        while charset:
            probs, h = net.predict(char, h, cuda=CUDA)
            # keep only unused letters, "remove" prior of letter from probabilities
            #TODO: wrong - normalize
            probs = {k: v / letter_priors[k] for k, v in probs.items() if k in charset + [" "]}
            char = np.random.choice(
                list(probs.keys()),
                p=np.array(list(probs.values())) / sum(probs.values()),
            )
            logprob += np.log(probs[char])
            if not char.isalpha():
                if last_word and not is_word(last_word):
                    break
                else:
                    last_word = ""
            else:
                last_word += char
            sentence += char
            if char not in (" ", "\n") or char in charset:
                charset.remove(char)
            else:
                if len([c for c in charset if c == " "]) > len(charset):
                    break  # avoid infinite loop
        
        if not charset and ((not last_word) or (is_word(last_word))):
            #TODO: also try permutations of found words
            if not best_attempt or logprob > best_attempt[1]:
                print("New best:", sentence, logprob)
                best_attempt = (sentence, logprob)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m anagram_solver <chars>")
        exit(1)
    chars = sys.argv[1]
    solve_anagram([c for c in chars])
