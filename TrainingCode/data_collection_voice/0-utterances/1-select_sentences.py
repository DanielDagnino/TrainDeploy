import argparse
import json
import sys

import spacy
from path import Path
from tqdm import tqdm

# Load the English language model in spaCy
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def is_there_a_name(_text):
    doc = nlp(_text)
    for token in doc:
        if token.ent_type_ in [
            'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
            'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'
        ]:
            return True
    return False


def phonetic_richness(_text):
    phonetic_units = {"a", "e", "i", "o", "u", "y", "ai", "au", "aw", "ay", "ea", "ee", "ei", "eu", "ew", "ey", "ie",
                      "oa", "oe", "oi", "oo", "ou", "oy", "ch", "sh", "th", "ph", "wh", "gh", "ng", "qu", "ck", "ar",
                      "er", "ir", "or", "ur"}

    _text = _text.lower()
    current_index = 0
    found_units = set()
    while current_index < len(_text):
        for size in range(2, 0, -1):
            unit = _text[current_index:current_index + size]
            if unit in phonetic_units:
                found_units.add(unit)
                current_index += size
                break
        current_index += 1

    richness = len(found_units)
    return richness


def run(dir_librispeech, dir_voice_cloners, subfolder, n_samples):
    richness_min = 10  # 10
    length_words_min_max = [20, 40]  # [15, 20]
    dataset_dir = Path(dir_librispeech)
    base_search_dir = dataset_dir / subfolder
    good_samples = []
    files = list(base_search_dir.walkfiles("*.txt"))
    print(f'len(files) = {len(files)}')
    for idx, fn in tqdm(enumerate(files), total=len(files)):
        samples = open(fn).readlines()
        for sample in samples:
            line = sample.strip().lower().split()
            idd, text = line[0], line[1:]
            len_text = len(text)
            text = " ".join(text)
            if length_words_min_max[0] < len_text < length_words_min_max[1] and not is_there_a_name(text):
                richness = phonetic_richness(text) if richness_min > 0 else True
                if richness > richness_min:
                    fn_text_relative = Path(fn).relpath(dataset_dir)
                    fn_relative_relative = fn_text_relative.parent / f'{idd}.flac'
                    voice_id = fn_text_relative.split("/")[1]
                    good_samples.append({
                        "utterance": text,
                        "voice_id": voice_id,
                        "fn_text": fn_text_relative,
                        "fn_audio": fn_relative_relative,
                        "dataset_dir": dataset_dir,
                        "dataset": "LibriSpeech",
                    })
                    if (len(good_samples) + 1) % 100 == 0:
                        print(f'len(good_samples) = {len(good_samples)}')
        if len(good_samples) >= n_samples:
            break

    Path(dir_voice_cloners).makedirs_p()
    json.dump(
        good_samples,
        open(
            f"{dir_voice_cloners}/samples_from_LibriSpeech"
            f'utterances-{subfolder.replace("-", "_")}-n_samples={len(good_samples)}-'
            f'words_min={length_words_min_max[0]}-words_max={length_words_min_max[1]}-richness_min={richness_min}.json',
            "w"), indent=4)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dir_librispeech", type=Path, help="Directory with the LibriSpeech dataset")
    parser.add_argument("dir_voice_cloners", type=Path, help="Output directory")
    parser.add_argument("--subfolder", type=Path, default="train-clean-100", help="Folder of LibriSpeech to use")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
