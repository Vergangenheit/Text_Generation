from data_loader.load import load_data
from configs import config
import os


def read_in_chunks(filename: str, batch_size: int, seq_length: int):
    with open(filename) as f:
        while True:
            chunk = f.read(batch_size * seq_length)
            if not chunk:
                break
            yield chunk


def generate(filename: str, batch_size: int, seq_length: int):
    for chunk in read_in_chunks(filename, batch_size, seq_length):
        X, y = load_data(chunk, seq_length)

        yield X, y


if __name__ == "__main__":
    data_gen = generate(os.path.join(config.PATH, config.TRAIN_FILE), 128, 50)
