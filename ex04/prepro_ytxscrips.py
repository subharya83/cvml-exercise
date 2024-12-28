"""
Downloads and tokenizes the YouTube transcripts (if available) given a channel id.
- The download is from YouTube.
- The tokenization is GPT-2 tokenizer with tiktoken

"""
import argparse
import os
import tiktoken
import numpy as np
import shutil
from deepmultilingualpunctuation import PunctuationModel as pm

DATA_CACHE_DIR = "data"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})


def tokenize(data_dir=None, out_dir=None, use_punctuation=False):
    _pm = None
    if use_punctuation:
        _pm = pm()

    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    print(f"Generating dataset from txt files in {data_dir}")
    # Input/Output initialization
    os.makedirs(out_dir, exist_ok=True)
    tmp_data = os.path.join(out_dir, 'data.txt')
    with open(tmp_data, 'wb') as outfile:
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'rb') as readfile:
                if use_punctuation:
                    try:
                        print(f'Punctuating file contents from {filepath} and writing to {tmp_data}')
                        contents = open(filepath, 'r').read()
                        result = _pm.restore_punctuation(contents)
                        result = result.replace(".", ".\r\n")
                        result = str.encode(result)
                        outfile.write(result)
                    except:
                        print(f'Falling back to regular file copy {filepath} as is to {tmp_data}')
                        shutil.copyfileobj(readfile, outfile)
                else:
                    print(f'Copying file {filepath} as is to {tmp_data}')
                    shutil.copyfileobj(readfile, outfile)

    train_path = os.path.join(out_dir, 'train.bin')
    val_path = os.path.join(out_dir, 'val.bin')

    text = open(tmp_data, 'r').read()
    text = "<|endoftext|>" + text
    text = text.replace('\n\n', '\n\n<|endoftext|>')
    # encode the text
    tokens = encode(text)
    tokens_np = np.array(tokens, dtype=np.int32)
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens_np = tokens_np[:32768]
    train_tokens_np = tokens_np[32768:]
    # save to file

    with open(val_path, "wb") as f:
        f.write(val_tokens_np.tobytes())
    with open(train_path, "wb") as f:
        f.write(train_tokens_np.tobytes())
    # prints
    print(f"Saved {len(val_tokens_np)} tokens to {val_path}")
    print(f"Saved {len(train_tokens_np)} tokens to {train_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help="Input directory containing txt files", required=True)
    parser.add_argument('-o', type=str, help="Path to store preprocessed data", required=True)
    parser.add_argument('-p', action='store_true', help="Use this to punctuate text")

    args = parser.parse_args()
    tokenize(data_dir=args.i, out_dir=args.o, use_punctuation=args.p)
