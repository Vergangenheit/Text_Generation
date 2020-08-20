import tensorflow as tf
from model.base_model import BaseModel
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from data_loader.load import load_final


class Lstm_Text_Gen(BaseModel):
    def __init__(self, config, seq_length, vocab_size):
        super().__init__(config)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.model = None
        self.X = None
        self.y = None

    def load_data(self):
        self.X, self.y = load_final(self.config.data.train_file)



    def build(self):
        inputs = Input(shape=(self.seq_length, self.vocab_size))
        embed = Embedding(input_dim=self.vocab_size, output_dim=self.config.model.embedding_dim,
                          input_length=self.seq_length)(inputs)
        lstm1 = Bidirectional(LSTM(self.config.model.lstm1))(embed)
        dropout1 = Dropout(self.config.model.dropout1)(embed)
        dense1 = Dense(self.config.model.dense1, activation='relu')(dropout1)
        output = Dense(self.vocab_size, activation='softmax')(dense1)

        self.model = Model(inputs=inputs, outputs=output)

    def train(self):
        self.model.compile(loss=self.config.train.loss, metrics=self.config.train.loss,
                           optimizer=self.config.train.optimizer)
        checkpoint = ModelCheckpoint(self.config.train.model_checkpoint, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, period=10)
        callback_list = [checkpoint]
        self.model.fit(self.X, self.y, epochs=self.config.train.epochs, batch_size=self.config.train.batch_size,
                       verbose=2, callbacks=callback_list)
    
    def evaluate(self):
        pass
    
    


