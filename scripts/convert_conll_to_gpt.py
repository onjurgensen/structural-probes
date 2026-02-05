import pandas as pd
import torch
import transformer_lens
from argparse import ArgumentParser
from conllu import parse_incr
import h5py
import numpy as np
import re
from tqdm import tqdm

class ConllToGPTConverter:
    def __init__(self, gpt_model):
        try:
            self.model = transformer_lens.HookedTransformer.from_pretrained(gpt_model)
            self.tokenizer = self.model.tokenizer
        except Exception:
            raise ValueError("selected gpt_model not found")
        self.layer_names = ['blocks.0.hook_resid_pre'] + [f'blocks.{i}.hook_resid_post' for i in range(self.model.cfg.n_layers)] # changed from 'hook_embed'
        self.feature_count = self.model.cfg.d_model

    def combine_token_embeddings(self, text, word_list, pooling): 
        tokenizer = self.tokenizer
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        offsets = encoding["offset_mapping"][0].tolist()
        _, cache = self.model.run_with_cache([text], prepend_bos=False)
        activations = {layer: cache[layer][0] for layer in self.layer_names}
        assert activations[self.layer_names[0]].shape[0] == len(offsets)
        word_spans = []
        cursor = 0
        for word in word_list:
            match = re.search(re.escape(word), text[cursor:])
            if match is None:
                raise ValueError(f"Word '{word}' not found after position {cursor}")
            start = cursor + match.start()
            end = start + len(word)
            word_spans.append((start, end))
            cursor = end
        word_embeddings = []
        token_idxs_list = []
        for (ws, we) in word_spans:
            token_idxs = [
                i for i, (ts, te) in enumerate(offsets)
                if ts < we and te > ws
            ]
            if not token_idxs:
                raise ValueError(
                    f"No tokens aligned to '{text[ws:we]}' at span {(ws, we)}"
                )
            token_embs = {layer: activations[layer][token_idxs] for layer in self.layer_names}
            if pooling == "last":
                word_emb = {layer: token_embs[layer][-1] for layer in self.layer_names}
                token_idxs = token_idxs[-1]
            elif pooling == "first":
                word_emb = {layer: token_embs[layer][0] for layer in self.layer_names}
                token_idxs = token_idxs[0]
            elif pooling == "mean":
                word_emb = {layer: token_embs[layer].mean(dim=0) for layer in self.layer_names}
                token_idxs = np.mean(token_idxs)
            word_embeddings.append(word_emb)
            token_idxs_list.append(token_idxs)
        stacked = torch.stack(
            [torch.stack([we[layer] for we in word_embeddings]) for layer in self.layer_names]
        )
        return stacked, token_idxs_list

    def convert(self, input_path, output_path, pkl_path):
        with h5py.File(output_path, 'w') as fout:
            data = []
            with open(input_path, "r", encoding="utf-8") as fin:
                sentences = list(parse_incr(fin))
                for idx, sent in enumerate(tqdm(sentences)):
                    text = sent.metadata["text"] if "text" in sent.metadata else " ".join([tok["form"] for tok in sent])
                    word_list = [tok["form"] for tok in sent]
                    stacked, token_idxs_list = self.combine_token_embeddings(text, word_list, pooling="mean")
                    fout.create_dataset(str(idx), data=stacked.cpu().numpy())
                    data.append({'sentence_idx': idx, 'word_list': word_list, 'token_idxs': token_idxs_list})
            pd.DataFrame(data).to_pickle(pkl_path)
def main():
    argp = ArgumentParser()
    argp.add_argument('input_path')
    argp.add_argument('output_path')
    argp.add_argument('gpt_model', help='gpt2, distilgpt2, gpt2-medium, gpt2-large, gpt2-xl')
    args = argp.parse_args()
    converter = ConllToGPTConverter(args.gpt_model)
    converter.convert(args.input_path, args.output_path, args.output_path.split(".hdf5", 1)[0] + "_toks.pkl")

if __name__ == "__main__":
    main()
        
        