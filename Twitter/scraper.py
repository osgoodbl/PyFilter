#! /usr/bin/python3

import settings
import tweepy
import sys
import unicodecsv as csv
import tweepy
import sys
import unicodecsv as csv

class CustomStreamListener(tweepy.StreamListener):

    def on_status(self, status): # if tweet is a retweet, contains my name or has futbol it in, skip the tweet
        if hasattr(status,'retweeted_status') or status.author.screen_name == 'tf_and_huracans' or 'futbol' in status.author.screen_name:
            return True
        elif hasattr(status,'extended_tweet'): #if tweet has "extended_tweet" attribute, process "extended_tweet" 
            print('New Extended Entitity Tweet')
            # Writing status data to csv
            with open('huracan.csv', 'ab') as f:
                writer = csv.writer(f)
                writer.writerow([str(status.author.screen_name), str(status.id_str), status.created_at, str(status.text), str(status.entities), str(status.extended_tweet)])
        else:
            print('New Entity Tweet')  # if tweet is a standard tweet, process tweet
            # Writing status data to csv
            with open('huracan.csv', 'ab') as f:
                writer = csv.writer(f)
                writer.writerow([str(status.author.screen_name), str(status.id_str), status.created_at, str(status.text), str(status.entities), str({"No Extended Entities":"Empty"}) ])

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True  # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True  # Don't kill the stream

    # Writing csv titles
    # This creates and writes headers to a csv file
with open('huracan.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Author', 'Id', 'Date', 'Text', 'Entities', 'Extended Entities'])

# connect to Twitter Streaming API
auth = tweepy.OAuthHandler(settings.consumer_key, settings.consumer_secret)
auth.set_access_token(settings.access_token, settings.access_token_secret)
api = tweepy.API(auth,timeout=200)

# instantiate the Twitter Stream API Listener
stream_listener = CustomStreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=settings.TRACK_TERMS,)
