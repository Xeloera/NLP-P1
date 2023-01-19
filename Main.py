import tweepy
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import time
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
        sentiment = "NEUTRAL" # replace this with your own sentiment analysis code
        hashtags = [hashtag['text'] for hashtag in tweet.entities['hashtags']]
        media_url = None #replace this with your own media extraction code
        if 'media' in tweet.entities:
            media_url = tweet.entities['media'][0]['media_url']
        return {'tweet_id': tweet.id,
                'created_at': created_at,
                'screen_name': screen_name,
                'location': location,
                'followers_count': followers_count,
                'retweet_count': retweet_count,
                'favorite_count': favorite_count,
                'sentiment': sentiment,
                'hashtags': hashtags,
                'media_url': media_url,
                'tokens': stemmed_tokens}
    def store_tweet_tokens(self, tweets, db_name):
        # Connect to SQLite database
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # Create table to store tweet tokens
        c.execute('''CREATE TABLE IF NOT EXISTS tweet_tokens
                    (tweet_id INTEGER PRIMARY KEY, created_at TIMESTAMP, user_name TEXT, user_location TEXT, user_followers INTEGER, retweet_count INTEGER, favorite_count INTEGER, sentiment TEXT, tokens TEXT, hashtags TEXT)'''
                )

        # Apply tokenization and stemming to tweets
        stemmer = SnowballStemmer("english")
        for tweet in tweets:
            # Tokenize tweet text
            tokens = word_tokenize(tweet.full_text)
            # Apply stemming to tokens
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            # Extract Media
            media_list = tweet.entities.get("media", [])
            # Extract Hashtags
            hashtags = tweet.entities.get("hashtags", [])
            if len(hashtags) > 0:
                hashtags = [hashtag["text"] for hashtag in hashtags]
            # Extract Sentiment
            #create an instance of the SentimentAnalyser
           # sentiment_analyzer = SentimentAnalyser(db_name)
           # #train the model
           # sentiment_analyzer.train_model()
           # #call the predict_sentiment method on the instance of the class
           # sentiment = sentiment_analyzer.predict_sentiment(stemmed_tokens)
           
            # Save tokens to SQLite database
            c.execute("INSERT INTO tweet_tokens VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (tweet.id, tweet.created_at, tweet.user.screen_name,
                    tweet.user.location, tweet.user.followers_count,
                    tweet.retweet_count, tweet.favorite_count, "lol",
                    ' '.join(stemmed_tokens), hashtags))
            conn.commit()

        # Close SQLite connection
        conn.close()

#' '.join(stemmed_tokens), 'tweet_tokens.db'
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd

class SentimentAnalyser:

    def __init__(self, db_name):
        self.db_name = db_name
        self.model = None
        self.tokenizer = None

    def train_model(self):
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("SELECT tokens, sentiment FROM tweet_tokens")
        rows = c.fetchall()
        conn.close()

        # Create dataframe from SQLite data
        data = pd.DataFrame(rows, columns=['tokens', 'sentiment'])

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(data['tokens'], data['sentiment'], test_size=0.01)

        # Tokenize tweets
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X_train)
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        # Pad sequences to have the same length
        max_length = max([len(x) for x in X_train + X_test])
        X_train = pad_sequences(X_train, maxlen=max_length)
        X_test = pad_sequences(X_test, maxlen=max_length)

        # Create LSTM model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
        self.model.add(LSTM(units=32))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train model
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

    def predict_sentiment(self, tweet):
        if self.model is None or self.tokenizer is None:
            print("Model not trained. Run train_model() first.")
            return None
        tweet = self.tokenizer.texts_to_sequences([tweet])
        tweet = pad_sequences(tweet, maxlen=self.model.input_shape[1])
        sentiment = self.model.predict(tweet)[0]
        sentiment = "Positive" if sentiment > 0.5 else "Negative"
        return sentiment




tp = TweetProcessor(consumer_key, consumer_secret, access_token,
                    access_token_secret)
trend = 'NLP'
tweets = tp.search_tweets(trend, 1)
for tweet in tweets:
    tweet_info = tp.preprocess_tweet(tweet)
    tp.store_tweet_tokens(tweets, 'tweet_tokens.db')
