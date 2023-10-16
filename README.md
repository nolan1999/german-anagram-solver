# german-anagram-solver

## Motivation
Why try it by hand, if you can use deep learning (and require 100x the time...).
This repo serves the sole purpose of solving a specific anagram in German, with spaces not included in the input.
Since I did not find a workable solution online for my specific problem (sentence too long for exhaustive search), I decided to try to find a solution.

## How it works  
A word sequence containing the input characters and a free number of spaces is repeatedly sampled
- the next character is sampled among those remaining (or space) according to the probability assigned to it by the language model, normalized by its prior probability (to favour less frequent characters)
- if the character is a space, a new word is created, that is checked against a list of German words; if it is not present, the process restarts since the sentence can not mean anything
- if all characters have been used, the cumulative probability assigned by the language model is checked against the current best
    - if the probability is higher, the newly found word is printed to the console for the user to evaluate
    - else, nothing happens
- a new word is sampled

## Does it work?
Not that well, no...

## How to use
1. Download training dataset: `python -m src.data.download_data`
2. Merge training dataset: `python -m src.data.merge_data <num_texts_to_use>`
3. Prepare char-int mapping: `python -m src.data.chars2int`
4. Prepare letter priors: `python -m src.data.letter_priors`
5. Run training notebook (`src/model/model.ipynb`)
6. Run anagram solver `python -m src.anagram_solver.anagram_solver_probabilistic <characters>`
    (other versions are deprecated).
