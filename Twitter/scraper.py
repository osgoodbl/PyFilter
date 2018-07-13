#! /Users/brianosgood/.virtualenvs/huracan/bin/python

import settings
import tweepy
import sys
import csv


class CustomStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if hasattr(status,'retweeted_status') or status.author.screen_name == 'tf_and_huracans' or 'futbol' in status.author.screen_name:
            return True
        elif hasattr(status,'extended_tweet'):
            print(status.author.screen_name, status.created_at, status.text)
            # Writing status data
            with open('huracan.csv', 'a') as f:
                writer = csv.writer(f, delimiter = str(","))
                writer.writerow([status.author.screen_name, status.id_str, status.created_at, str(status.text), str(status.entities), str(status.extended_tweet)])
        else:
            print(status.author.screen_name, status.created_at, status.text)
            # Writing status data
            with open('huracan.csv', 'a') as f:
                writer = csv.writer(f, delimiter = str(","))
                writer.writerow([status.author.screen_name, status.id_str, status.created_at, str(status.text), str(status.entities), {"No Extended Entities":"Empty"} ])

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True  # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True  # Don't kill the stream

    # Writing csv titles


with open('huracan.csv', 'w') as f:
    writer = csv.writer(f, delimiter = str(","))
    writer.writerow(['Author', 'Id', 'Date', 'Text', 'Entities', 'Extended Entities'])


auth = tweepy.OAuthHandler(settings.consumer_key, settings.consumer_secret)
auth.set_access_token(settings.access_token, settings.access_token_secret)
api = tweepy.API(auth,timeout=200)

stream_listener = CustomStreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=settings.TRACK_TERMS, encoding='utf-8')

