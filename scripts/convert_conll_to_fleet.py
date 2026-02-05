import os
import pickle
from transformers import AutoTokenizer
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

import re
import h5py
from tqdm import tqdm
from argparse import ArgumentParser
from conllu import parse_incr
import sys
import importlib
import numpy as np

load_dotenv()

# Import GPT at module level
sys.path.insert(0, "./fmt-analysis")
from model_HF import GPT

class ConllToFleetConverter:
    def __init__(self, model_name, hf_repo_root):
        self.hf_repo_root = hf_repo_root
        self.model = GPT.from_pretrained(f"{hf_repo_root}/{model_name}", use_safetensors=False)  
        self.tokenizer = self.load_tokenizer(hf_repo_root, "babylm_full_bpe_100M_8k") #Since this model required the 100M tokenizer
        self.model.eval()
        self.model.to("cuda")


    ##Set Tokenizer Root as the folder that contains different tokenizer folders  (Same instructions as before)
    # Options - (relevant ones) - 
    #   babylm_full_bpe_8k - Tokenizer for 10M models, vocab size 8k
    #   babylm_full_bpe_100M_8k - Tokenizer for 100M models, vocab size 8k 
    #Model's Relevant details can be found in the Model Table in the database (in rundata.xlsx) 

    def load_tokenizer(self, hf_repo_root, data_dir = "babylm_full_bpe_100M_8k"):
        """
        Load tokenizer for natural stories evaluation.

        Args:
            data_dir (str): The directory path where the tokenizer data is stored.

        Returns:
            tokenizer (Tokenizer): The loaded tokenizer object.

        Raises:
            NotImplementedError: If stoi/itos is not supported or found.

        """
        # data_dir = os.path.join(hf_repo_root, "data", data_dir)
        if hf_repo_root == "fmtmodels":
            data_dir = os.path.join("./fmt-analysis", "data", data_dir)
        else:
            data_dir = os.path.join(hf_repo_root, "data", data_dir)

        meta_path = os.path.join(data_dir, "meta.pkl")
        load_meta = os.path.exists(meta_path)

        if load_meta:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            if meta.get("custom_tokenizer", False):
                print(f"Loading custom tokenizer from {data_dir}")
                tokenizer = AutoTokenizer.from_pretrained(data_dir, use_fast=False)
            else:
                if meta.get("stoi", False):
                    raise NotImplementedError("stoi/itos not supported yet")
                else:
                    raise NotImplementedError("No stoi/itos found")
        else:
            print("No meta.pkl found")
            raise NotImplementedError("No meta.pkl found")

        if not tokenizer.eos_token:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"  # Add if needed?
        return tokenizer

    def combine_token_embeddings(self, text, word_list, pooling): 
        tokenizer = self.tokenizer
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        offsets = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs_m1 = self.model(encoding['input_ids'].to("cuda"), hidden_states=True)

        activations = {i: outputs_m1["hidden_states"][i][0] for i in range(len(outputs_m1["hidden_states"]))}
        
        assert activations[list(activations.keys())[0]].shape[0] == len(offsets)
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
            token_embs = {layer: activations[layer][token_idxs] for layer in activations.keys()}
            if pooling == "last":
                word_emb = {layer: token_embs[layer][-1] for layer in activations.keys()}
                token_idxs = token_idxs[-1]
            elif pooling == "first":
                word_emb = {layer: token_embs[layer][0] for layer in activations.keys()}
                token_idxs = token_idxs[0]
            elif pooling == "mean":
                word_emb = {layer: token_embs[layer].mean(dim=0) for layer in activations.keys()}
                token_idxs = np.mean(token_idxs)
            word_embeddings.append(word_emb)
            token_idxs_list.append(token_idxs)
        stacked = torch.stack(
            [torch.stack([we[layer] for we in word_embeddings]) for layer in activations.keys()],
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
    argp.add_argument('model_name')
    argp.add_argument('hf_repo_root')
    args = argp.parse_args()

    # if args.hf_repo_root not in sys.path:
    #     sys.path.insert(0, args.hf_repo_root)
    # GPT = importlib.import_module("model_HF").GPT

    hf_login(token=os.environ.get("HF_ACCESS_TOKEN"))
    converter = ConllToFleetConverter(args.model_name, args.hf_repo_root)
    converter.convert(args.input_path, args.output_path, args.output_path.split(".hdf5", 1)[0] + "_toks.pkl")

if __name__ == "__main__":
    main()