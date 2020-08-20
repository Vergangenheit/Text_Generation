import os

PATH = '/content/drive/My Drive/Text_Generation_LSTM'
TRAIN_FILE = 'wikitext-103-raw/wiki.train.raw'
VALID_FILE = 'wikitext-103-raw/wiki.valid.raw'
TOKENIZER_FILE = 'tokenizer.pkl'
SAVE_PATH = '/content/drive/My Drive/Text_Generation_LSTM/ckpt_republic_model'
MODEL_CKPT = os.path.join(SAVE_PATH, 'model_republic_epoch_{epoch:03d}')

CFG = {
    "data": {
        "train_file": TRAIN_FILE,
        "valid_file": VALID_FILE
    },
    "train": {
        "loss": 'categorical_crossentropy',
        "metrics": ["accuracy"],
        "optimizer": "adam",
        "batch_size": 128,
        "epochs": 100,
        "model_checkpoint": MODEL_CKPT

    },
    "model": {
        "embedding_dim": 50,
        "lstm1": 128,
        "dropout1": 0.2,
        "dense1": 100
    }
}