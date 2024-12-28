from paddleocr import PaddleOCR
from trankit import Pipeline
from img2vec_pytorch import Img2Vec
import os
from PIL import Image
import argparse


def process_images(input_dir=None, output_dir=None):
    if os.path.exists(input_dir):
        # Loading image based models
        _ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=True)
        #_i2v = Img2Vec(cuda=True)

        _l = os.listdir(input_dir)
        for _f in _l:
            _img = os.path.join(args.i, _f)
            # img = Image.open(_img)
            # Get a vector from img2vec, returned as a torch FloatTensor
            # vec = _i2v.get_vec(img, tensor=False).tolist()
            # print(vec)
            _result = _ocr.ocr(_img)
            # Output line by line text detection
            for _elems in _result[0]:
                # Convert co-ordinates into string for future feature analysis
                _str = [int(x) for xs in _elems[0] for x in xs]
                print(_f, _str, _elems[1][0])


def process_tokens(input_file=None):
    if os.path.exists(input_file):
        p = Pipeline(lang='english', gpu=True, cache_dir='./cache')
        with open(input_file, 'r', encoding='UTF-8') as file:
            while line := file.readline():
                _doc = line.rstrip()
                _tdoc = p.ner(_doc, is_sent=True)
                print(_tdoc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform OCR on a directory containing images')
    parser.add_argument('-i', type=str, help='Input', required=True)
    parser.add_argument('-m', type=int, help='Mode', required=True)

    args = parser.parse_args()
    if int(args.m) == 0:
        # Run OCR, get embeddings
        process_images(input_dir=args.i)
    if int(args.m) == 1:
        # Run NLP
        process_tokens(input_file=args.i)

