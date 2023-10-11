import os
import sys


N_TEXTS = 1000


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_texts = int(sys.argv[1])
    else:
        n_texts = N_TEXTS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(
        script_dir,
        "download",
        "Corpus of German-Language Fiction",
        "corpus-of-german-fiction-txt",
    )
    output_file = os.path.join(script_dir, "corpus.txt")

    with open(output_file, "w") as outfile:
        filenames = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        for filename in filenames[:n_texts]:
            with open(os.path.join(folder_path, filename), "r") as infile:
                outfile.write(infile.read())
                outfile.write("\n")
