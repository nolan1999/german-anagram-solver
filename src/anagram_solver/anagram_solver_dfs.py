import copy
import numpy as np
import os
import sys
import torch
from dataclasses import dataclass, field
from spellchecker import SpellChecker
from src.model.model import CharRNN


CUDA = False  #torch.cuda.is_available()
BEST_N = 5


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


class PartialAnagram:
    def __init__(self, char, h, sentence, remaining, last_word, neglogprob):
        self.char = char
        self.h = h
        self.sentence = sentence
        self.remaining = remaining
        self.last_word = last_word
        self.neglogprob = neglogprob


def solve_anagram(net, chars, best_n=BEST_N, cuda=CUDA, lang="de"):
    assert lang == "de"
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "words.txt"), "r") as f:
        words = set([w.strip().lower() for w in f.readlines()])
    # spell = SpellChecker(language=lang)
    # is_word = lambda w: bool(spell.known([w])) if len(w) > 2 else w in words  # German only
    is_word = lambda w: w.lower() in words

    best_attempts = []  # Stores the best results
    queue = []  # Stores the partial anagrams

    # Initialize the queue
    queue.append(PartialAnagram(
        char="\n",
        h=net.init_hidden(1),
        sentence="",
        remaining=copy.deepcopy(list(chars)),
        last_word="",
        neglogprob=0,
    ))

    while queue:
        el = queue.pop()
        neglogprob = el.neglogprob

        # Last word handling
        if not el.char.isalpha():
            if el.last_word and not is_word(el.last_word):
                continue
            else:
                el.last_word = ""
        else:
            el.last_word += el.char

        # Add next char
        el.sentence += el.char
        if el.char not in (" ", "\n"):
            el.remaining.remove(el.char)

        # Finished letters
        if len(el.remaining) == 0:
            if el.last_word and is_word(el.last_word):
                print(f"Found {el.sentence} ({neglogprob})")
                best_attempts.append((el.sentence, neglogprob))
                best_attempts.sort(key=lambda x: x[1], reverse=False)
                best_attempts = best_attempts[:best_n]
            continue

        # Next character prediction
        probs, h = net.predict(el.char, el.h, cuda=cuda)
        probs = {k: v for k, v in probs.items() if k in el.remaining + [" "]}
        # Append highest probability last
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=False)

        for next_char, next_prob in sorted_probs:
            if el.char in (" ", "\n") and next_char in (" ", "\n"):
                continue  # avoid infinite loop
            queue.append(
                PartialAnagram(
                    char=next_char,
                    h=h,
                    sentence=el.sentence,
                    remaining=copy.deepcopy(el.remaining),
                    last_word=el.last_word,
                    neglogprob=neglogprob - np.log(next_prob),
                )
            )

    return {k: v for k, v in best_attempts}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m anagram_solver <chars>")
        exit(1)
    chars = [c for c in sys.argv[1]]
    for sentence, logprob in solve_anagram(net, chars).items():
        print(f"{sentence} ({logprob})")
