import json
import os


if __name__ == "__main__":
    script_path = os.path.dirname(__file__)
    corpus_path = os.path.join(script_path, "corpus.txt")
    with open(corpus_path, 'r') as f:
        text = f.read()

    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    with open(os.path.join(script_path, 'int2char.json'), 'w') as f:
        json.dump(int2char, f)

    with open(os.path.join(script_path, 'char2int.json'), 'w') as f:
        json.dump(char2int, f)
