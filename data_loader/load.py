import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from pickle import dump
import os
import numpy as np
from configs import config
import string


def load_doc(filename: str) -> str:
    """load text file into memory
    Args: filename
    Returns: string (text)"""
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def clean_doc(doc: str) -> list:
    """turns a doc into clean tokens
    Args: doc(text)
    Returns: list of tokens"""
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


def create_seq(seq_length: int, tokens: list) -> list:
    """organize tokens into sequences of text
    Args: seq_length(int), list of tokens(words)
    Returns: list containing sequences"""
    length = seq_length + 1
    sequences = []
    for i in range(length, len(tokens)):
        # select a sequence of tokens
        seq = tokens[i - length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print("Total sequences is %d" % len(sequences))

    return sequences


def save_doc(lines, filename):
    """saves sequences into a file
    Args: list, filename(str)
    Returns: None"""
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_sequences(filename: str) -> list:
    """reads sequence file into a list
    Args: filename(str)
    returns: list
    """
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    lines = text.split('/n')

    return lines


def encode_sequences(lines: list) -> (list, int):
    """encodes sequences into arrays of tokens
    Args: sequences(list)
    Returns: tokenized sequences(list) and vocab size(int)"""
    tk = Tokenizer()
    tk.fit_on_texts(lines)
    sequences = tk.texts_to_sequences(lines)
    dump(tk, open(os.path.join(config.PATH, config.TOKENIZER_FILE), 'wb'))
    # vocabulary size
    vocab_size = len(tk.word_index) + 1
    print("Vocabulary size is %d" % vocab_size)

    return sequences, vocab_size


def inputs_outputs(sequences: list, vocab_size: int) -> (np.array, np.array):
    """creates features and labels by seprating sequences' last token
    Args: sequences(list)
    Returns: features and labels (arrays)"""
    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    # seq_length = X.shape[1]

    return X, y


def load_final(filename):
    """puts all preprocessing functions together
    Args: data filename(str)
    Returns: features and labels arrays"""
    text = load_doc(filename)
    tokens = clean_doc(text)
    sequences = create_seq(50, tokens)
    sequences, vocab_size = encode_sequences(sequences)
    X, y = inputs_outputs(sequences, vocab_size)

    return X, y
