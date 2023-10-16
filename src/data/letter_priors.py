import collections
import json
import numpy as np
import os


CORPUS_PATH = os.path.join(os.path.dirname(__file__), 'corpus.txt')
LETTER_PRIORS_PATH = os.path.join(os.path.dirname(__file__), 'letter_priors.json')


if __name__ == '__main__':
    with open(CORPUS_PATH, 'r') as f:
        text = f.read()
    letter_counts = collections.Counter(text.lower())
    letters = letter_counts.keys()
    counts = np.array(list(letter_counts.values()))
    probs = counts / np.sum(counts)
    probs = {l: float(p) for l, p in zip(letters, probs)}  # json-serializable
    with open(LETTER_PRIORS_PATH, 'w') as f:
        json.dump(probs, f)
