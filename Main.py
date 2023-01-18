from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Attention
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
import numpy as np
from tensorflow.python.framework import base as trackable
# Load the labeled tweets dataset
tweets = ["This is a positive tweet.",
          "This is a negative tweet.", "This is a neutral tweet."]
labels = [1, 0, 0]  # 1 for positive, 0 for negative or neutral

# Tokenize the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

# Pad the sequences
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Train a self-attention RNN model on the tokenized tweets
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) +
          1, output_dim=100, input_length=max_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Attention())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(padded_sequences, np.array(labels), epochs=10)

# Save the trained model
model.save("model/self_attention_rnn_model.h5")
