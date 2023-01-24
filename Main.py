import tweepy
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import time
import json
from keras.layers import *
import keras as K
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, TimeDistributed
from keras.models import Sequential
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input
import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#from keras_self_attention import Attention
#nltk.download('stopwords')
#nltk.download('punkt')
# Authenticate with Twitter API
consumer_key = '4SYgPZcw1LaT1P5pEykv7f5CA'
consumer_secret = 'EadeS0ZUmHCZB3VofDqhP0Uz9vrDpggHfsXitNAEMUUSceoywX'
access_token = '1320596574062419970-sk3xx9YRoTu0s7KzkGxRGMv45WYtuh'
access_token_secret = 'J7KCL0BqjIAhCBtJyYWaKG3yh7qRPs9Y5say862aPch1J'

class TweetProcessor:

    def __init__(self, consumer_key, consumer_secret, access_token,
                 access_token_secret):
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")

    def search_tweets(self, trend, num_tweets):
        tweets = []
        for tweet in tweepy.Cursor(self.api.search_tweets,
                                   q=trend,
                                   lang='en',
                                   tweet_mode='extended').items(num_tweets):
            time.sleep(1)
            tweets.append(tweet)
        return tweets

    def preprocess_tweet(self, tweet):
        # Tokenize tweet text
        tokens = word_tokenize(tweet.full_text)
        # Remove stop words
        filtered_tokens = [
            token for token in tokens if token.lower() not in self.stop_words
        ]
        # Apply stemming to tokens
        stemmed_tokens = [
            self.stemmer.stem(token) for token in filtered_tokens
        ]
        # Extract additional information
        created_at = tweet.created_at
        screen_name = tweet.user.screen_name
        location = tweet.user.location
        followers_count = tweet.user.followers_count
        retweet_count = tweet.retweet_count
        favorite_count = tweet.favorite_count
        full_text = tweet.full_text
        sentiment = "NEUTRAL" # replace this with your own sentiment analysis code
        hashtags = [hashtag['text'] for hashtag in tweet.entities['hashtags']]
        media_url = None #replace this with your own media extraction code
        if 'media' in tweet.entities:
            media_url = tweet.entities['media'][0]['media_url']
        return {
            'tweet_id': tweet.id,
            'created_at': created_at,
            'screen_name': screen_name,
            'location': location,
            'followers_count': followers_count,
            'retweet_count': retweet_count,
            'favorite_count': favorite_count,
            'sentiment': sentiment,
            'hashtags': hashtags,
            'media_url': media_url,
            'tokens': stemmed_tokens,
            'full_text': full_text
        }
    def store_tweet_tokens(self, tweets, db_name):
        # Connect to SQLite database
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # Create table to store tweet tokens
        c.execute('''CREATE TABLE IF NOT EXISTS tweet_tokens
                    (tweet_id INTEGER PRIMARY KEY,
                    created_at TIMESTAMP, 
                    user_name TEXT, 
                    user_location TEXT, 
                    user_followers INTEGER, 
                    retweet_count INTEGER, 
                    favorite_count INTEGER, 
                    sentiment TEXT, 
                    tokens TEXT, 
                    hashtags TEXT,
                    label INTEGER,
                    full_text TEXT)''')

        # Apply tokenization and stemming to tweets
        stemmer = SnowballStemmer("english")
        for tweet in tweets:
            # Tokenize tweet text
            tokens = word_tokenize(tweet.full_text)
            # Apply stemming to tokens
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            toto = ','.join(stemmed_tokens)
            # Extract Media
            media_list = tweet.entities.get("media", [])
            # Extract Hashtags
            hashtags = tweet.entities.get("hashtags", [])
            hashtags = json.dumps(hashtags)
            # Extract Sentiment
            sentiment = 10
            label = 10
            # Save tokens to SQLite database
            tweet_id = tweet.id
            c.execute("SELECT * FROM tweet_tokens WHERE tweet_id=?", (tweet_id,))
            data = c.fetchone()
            if data is None:
                c.execute("INSERT INTO tweet_tokens VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (tweet.id, 
                         tweet.created_at, 
                         tweet.user.screen_name, 
                         tweet.user.location, 
                         tweet.user.followers_count, 
                         tweet.retweet_count, 
                         tweet.favorite_count, 
                         sentiment, 
                         ' '.join(stemmed_tokens), 
                         hashtags,
                         label, 
                         tweet.full_text))
                conn.commit()

        # Close SQLite connection
        conn.close()
    def retrieve_and_save_tweet(tweet_id): #fix this
    # set up authentication for the Twitter API
        conn = sqlite3.connect('tweets.db')
        cursor = conn.cursor()
        # retrieve the tweet using its id
        tweet = self.api.get_status(tweet_id, tweet_mode='extended')

        # insert the tweet's text and id into the SQLite database
        cursor.execute("INSERT INTO tweets (id, text) VALUES (?, ?)", (tweet.id, tweet.full_text))
        conn.commit()
        print(f"Tweet with id {tweet.id} saved to the database.")

#' '.join(stemmed_tokens), 'tweet_tokens.db'
from keras.models import load_model
import pickle
# Attentional Layer
class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


from keras.utils import np_utils
#make me a nerual network that can be trained then used to create text using a transformer

class SentimentAnalyser:

    def __init__(self,
                 db_name,
                 model=None,
                 tokenizer=None,
                 filepath=None,
                 tokenizer_path=None):
        self.db_name = db_name
        if model != None:                                     #fix this so that the tokenizer path is init properly using the load fuctions below
            self.model = load_model(model)
        if tokenizer != None:
            with open(tokenizer, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
                self.tokenizer_path = filepath + "_tokenizer.pkl"
        else:
            self.model = None
            self.tokenizer = None
            self.tokenizer_path = None

    def load_model(self, filepath):
        self.model = load_model(filepath + '.h5')
        with open(filepath + "_tokenizer.pkl", 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.tokenizer_path = filepath+"_tokenizer.pkl"
        print("Model and tokenizer loaded from", filepath)

    def save_model(self, filepath):
        self.model.save(filepath + '.h5')
        with open(filepath + "_tokenizer.pkl", 'wb') as handle:
            pickle.dump(self.tokenizer,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        self.model = load_model(filepath + '.h5')
        self.tokenizer_path = filepath + "_tokenizer.pkl"
        print("Model and tokenizer saved to", filepath)
    def train_model(self):
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        # Retrieve text data from database
        c.execute("SELECT tokens FROM tweet_tokens")
        rows = c.fetchall()
        conn.close()
        # Convert tuple of text data to list of text documents
        text_data = [row[0] for row in rows]
        # Tokenize tweets
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(text_data)
        X_data = self.tokenizer.texts_to_sequences(text_data)
        # Pad sequences to have the same length
        max_length = max([len(x) for x in X_data])
        X_data = pad_sequences(X_data, maxlen=max_length)
        y_data = np.concatenate((X_data[:, 1:], np.zeros(
            (X_data.shape[0], 1))),
                                axis=1)

        print("X_data shape:", X_data.shape)
        print("y_data shape:", y_data.shape)
        # Create autoencoding self
        input_dim = len(self.tokenizer.word_index) + 1
        y_data = np_utils.to_categorical(y_data, num_classes=input_dim)
        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_length))
        self.model.add(LSTM(units=32, return_sequences=True))
        self.model.add(Bidirectional(LSTM(units=32)))
        self.model.add(
            TimeDistributed(Dense(units=input_dim, activation='softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Train model
        self.model.fit(X_data, y_data, epochs=3, batch_size=64)
        #Save Model
        self.save_model()
        self.model.save(f'{str(self.model)}.h5')
        with open(f'{self.tokenizer}.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def retrain_model(self, new_data):
        # Tokenize new data
        self.tokenizer.fit_on_texts(new_data)
        X_data = self.tokenizer.texts_to_sequences(new_data)

        # Pad sequences to have the same length
        max_length = max([len(x) for x in X_data])
        X_data = pad_sequences(X_data, maxlen=max_length)

        # Retrain model
        self.model.fit(X_data, X_data, epochs=3, batch_size=64)
        self.model.save(f'{str(self.model)}.h5')
        with open(f'{str(self.tokenizer)}.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    def predict_sentiment(self, tweet):
        if self.model is None or self.tokenizer is None:
            print("Model not trained. Run train_model() first.")
            return None
        tweet = self.tokenizer.texts_to_sequences([tweet])
        tweet = pad_sequences(tweet, maxlen=self.model.input_shape[1])
        sentiment = self.model.predict(tweet)[0]
        sentiment = "Positive" if sentiment > 0.5 else "Negative"
        return sentiment

    def generate_summary(self, trend): # mayber make this do that it can use a transformer as well as the self-attentional model so that you can generate two sumamries and combine them
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("SELECT full_text FROM tweet_tokens WHERE trend=?", (trend,))
        rows = c.fetchall()
        conn.close()
        # Tokenize tweets
        tweets = [row[0] for row in rows]
        tokens = self.tokenizer.texts_to_sequences(tweets)
        tokens = pad_sequences(tokens, maxlen=self.model.input_shape[1])

        # Generate summary using attention layer
        attention_layer = Attention()(self.model.layers[1].output)
        attention_model = Model(inputs=self.model.input, outputs=attention_layer)
        attention_vectors = attention_model.predict(tokens)
        attention_vectors = attention_vectors.reshape((attention_vectors.shape[0], attention_vectors.shape[2]))
        attention_vectors = attention_vectors / attention_vectors.sum(axis=1, keepdims=True)
        summary = ""
        for i in range(len(tweets)):
            weighted_tokens = [attention_vectors[i, j] * tokens[i][j] for j in range(len(tokens[i]))]
            summary += " ".join([self.tokenizer.index_word[index] for index in weighted_tokens if index != 0]) + " "
        return summary
    def update(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        # Connect to SQLite database
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        # Retrieve text data from database
        c.execute("SELECT tokens, tweet_id FROM tweet_tokens")
        rows = c.fetchall()
        conn.close()
        # Convert tuple of text data to list of text documents
        text_data = [row[0] for row in rows]
        ids = [row[1] for row in rows]
        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(text_data)

        # Use k-means to cluster the text data into different groups
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(X)

        # Assign labels to each text document based on the cluster they belong to
        labels = kmeans.labels_

        # Add labels column to database
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        for i in range(0,len(labels)):
            c.execute("UPDATE tweet_tokens SET label=? WHERE tweet_id=?",
                      (int(labels[i]), ids[i]))
        conn.commit()
        conn.close()
#make a function that generates a summary using self-attentional RNN layer and transformers to create a comprehensive summery from a collection of tokens

#tp = TweetProcessor(consumer_key, consumer_secret, access_token,
#                    access_token_secret)
#trend = 'Machine Learning'
#tweets = tp.search_tweets(trend, 50)
#for tweet in tweets:
#    tweet_info = tp.preprocess_tweet(tweet)
#    tp.store_tweet_tokens(tweets, 'tweet_tokens.db')

sa = SentimentAnalyser('tweet_tokens.db')
sa.update()
def generate_summary_torch():
    #import necessary libraries
    import torch
    from transformers import BertModel, BertTokenizer

    conn = sqlite3.connect("tweet_tokens.db")
    c = conn.cursor()
    c.execute("SELECT tokens FROM tweet_tokens")
    rows = c.fetchall()
    conn.close()
    tweets = [row[0] for row in rows]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    #instantiate the tokenizer and model

    #tokenize the input tokens
    # for tweet in tweets:
    #     tokenized_tokens = tokenizer.tokenize(tweets)

    #convert the tokens to their corresponding IDs in BERT's vocabulary
    indexed_tokens = tokenizer.convert_tokens_to_ids(tweets)

    #create a tensor from the indexed tokens
    tokens_tensor = torch.tensor([indexed_tokens])

    #pass the tensor through the model to get the self-attentional RNN layer output
    with torch.no_grad():
        outputs = model(tokens_tensor)

        #get the last hidden state of the RNN layer (this will be used as our summary)
        last_hidden = outputs[0][0][-1]

        #convert back to tokens using BERT's vocabulary and join them together into a summary string
        summary = ' '.join(tokenizer.convert_ids_to_tokens(last_hidden))

        return summary


import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value):
        # compute the attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.d_model)
        # apply softmax to the attention scores
        attn_weights = nn.Softmax(dim=-1)(attn_scores)
        # compute the weighted sum of the values using the attention scores
        output = torch.matmul(attn_weights, value)
        return output


class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # apply self-attention
        attn_output = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # apply feedforward
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x
class Encoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#            if len(hashtags) > 0:
#                hashtags = [hashtag["text"] for hashtag in hashtags]



'''
def generate_summary(tokens):
  #import necessary libraries
  import torch
  from transformers import BertModel, BertTokenizer

  #instantiate the tokenizer and model
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  #tokenize the input tokens
  tokenized_tokens = tokenizer.tokenize(tokens)

  #convert the tokens to their corresponding IDs in BERT's vocabulary
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_tokens)

  #create a tensor from the indexed tokens
  tokens_tensor = torch.tensor([indexed_tokens])

  #pass the tensor through the model to get the self-attentional RNN layer output
  with torch.no_grad():
    outputs = model(tokens_tensor)

    #get the last hidden state of the RNN layer (this will be used as our summary)
    last_hidden = outputs[0][0][-1]

    #convert back to tokens using BERT's vocabulary and join them together into a summary string
    summary = ' '.join(tokenizer.convert_ids_to_tokens(last_hidden))

    return summary
'''


'''
#    def train_model(self):
#        # Connect to SQLite database
#        conn = sqlite3.connect(self.db_name)
#        c = conn.cursor()
#        c.execute("SELECT tokens, sentiment FROM tweet_tokens")
#        rows = c.fetchall()
#        conn.close()
#
#        # Create dataframe from SQLite data
#        data = pd.DataFrame(rows, columns=['tokens', 'sentiment'])
#
#        # Split data into training and test sets
#        X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data['sentiment'], test_size=0.01)
#
#        # Tokenize tweets
#        self.tokenizer = Tokenizer()
#        self.tokenizer.fit_on_texts(X_train)
#        X_train = self.tokenizer.texts_to_sequences(X_train)
#        X_test = self.tokenizer.texts_to_sequences(X_test)
#
#        # Pad sequences to have the same length
#        max_length = max([len(x) for x in X_train + X_test])
#        X_train = pad_sequences(X_train, maxlen=max_length)
#        X_test = pad_sequences(X_test, maxlen=max_length)
#
#        # Create LSTM model
#        self.model = Sequential()
#        self.model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
#        self.model.add(LSTM(units=32, return_sequences=True))
#        self.model.add(Attention())
#        self.model.add(Dense(units=1, activation='sigmoid'))
#        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#        # Train model
#        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
#        #Save Model
#        self.model.save(f'{str(model)}.h5')
'''