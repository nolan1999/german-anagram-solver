# german-anagram-solver
Why try it by hand, if you can use deep learning (and need 100x the time...)

1. Download training dataset: `python -m src.data.download_data`
2. Merge training dataset: `python -m src.data.merge_data <num_texts_to_use>`
3. Prepare char-int mapping: `python -m src.data.chars2int`
4. Run training notebook (`src/model/model.ipynb`)
5. Run anagram solver `python -m src.anagram_solver.anagram_solver <characters>`
