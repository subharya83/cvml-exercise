import os
from collections import Counter
import json

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, caption_list):
        frequencies = Counter()
        idx = 4
        
        for caption in caption_list:
            for word in caption.split():
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        tokenized_text = text.split()
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]
    
    def decode(self, indices):
        return [self.itos[i] for i in indices]
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'itos': self.itos, 'stoi': self.stoi, 
                      'freq_threshold': self.freq_threshold}, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        vocab = cls(freq_threshold=data['freq_threshold'])
        vocab.itos = {int(k): v for k, v in data['itos'].items()}
        vocab.stoi = data['stoi']
        return vocab

def build_vocab_from_captions(captions_folder, threshold=5):
    vocab = Vocabulary(freq_threshold=threshold)
    captions = []
    
    for filename in os.listdir(captions_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(captions_folder, filename), 'r') as f:
                captions.extend(f.readlines())
    
    vocab.build_vocabulary([cap.strip() for cap in captions])
    return vocab