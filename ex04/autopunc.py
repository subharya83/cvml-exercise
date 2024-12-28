import argparse
import os
import shutil
from deepmultilingualpunctuation import PunctuationModel

def punc_dir(data_dir=None, out_dir=None):
    _pm = None
    # Input/Output initialization
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as readfile:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help="Input dir", required=True)
    parser.add_argument('-o', type=str, help="Output dir", required=True)
    args = parser.parse_args()
    punc_dir(input_file=args.i, output_file=args.o)
