import argparse
from conllu import parse_incr, TokenList

argp = argparse.ArgumentParser()
argp.add_argument('input_conll_filepath')
argp.add_argument('output_conll_filepath')
args = argp.parse_args()

cleaned_sentences = []

with open(args.input_conll_filepath, "r", encoding="utf-8") as fin:
    sentences = list(parse_incr(fin))
    for sentence in sentences:
        filtered_tokens = [token for token in sentence if isinstance(token["id"], int)]
        if filtered_tokens:
            cleaned_sentences.append(TokenList(filtered_tokens, sentence.metadata))

with open(args.output_conll_filepath, "w", encoding="utf-8") as fout:
    for sent in cleaned_sentences:
        fout.write(sent.serialize())
        # fout.write("\n")