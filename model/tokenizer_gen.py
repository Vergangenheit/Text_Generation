from tensorflow.keras.preprocessing.text import Tokenizer
from data_loader.load import create_seq, clean_doc
import os
from configs import config
from pickle import dump


def texts_gen(filename: str, chunk_size: int):
    with open(filename) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def text_generator(texts_generator, seq_length: int):
    for texts in texts_generator:
        tokens = clean_doc(texts)
        sequence = create_seq(seq_length, tokens)
        for seq in sequence:
            yield seq


def fit_tokenizer():
    texts_generator = texts_gen(os.path.join(config.PATH, config.TRAIN_FILE), 1000000)
    t_gen = text_generator(texts_generator)
    tk = Tokenizer()
    tk.fit_on_texts(t_gen)
    dump(tk, open(os.path.join(config.PATH, config.TOKENIZER_FILE), 'wb'))


if __name__ == "__main__":
    fit_tokenizer()
