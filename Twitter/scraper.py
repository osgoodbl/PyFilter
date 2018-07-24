#! /usr/bin/python3

import settings
import tweepy
import sys
import unicodecsv as csv
import json
import dataset
from sqlalchemy.exc import ProgrammingError
from flatten_json import flatten
import sqlite3
import os
import pandas as pd

db = dataset.connect("sqlite:///./tweets.db")



class CustomStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if hasattr(status,'retweeted_status') or status.author.screen_name == 'tf_and_lambos' or 'futbol' in status.author.screen_name:
            return True
        elif hasattr(status,'extended_tweet'):      
            print('New Extended Tweet')
            text = status.text
            name = status.user.screen_name
            id_str = status.id_str
            created = status.created_at
            entities = str(status.entities)
            ext_entities = str(status.extended_tweet)

            table = db["tweets"]
            try:
                table.insert(dict(
                    Text=text,
                    Author=name,
                    Id_str=id_str,
                    Date=created,
                    Entities = (entities),
                    Extended_Entities = (ext_entities),
                ))
            except ProgrammingError as err:
                print(err)
            
            
        else:
            print('New Regular Tweet')
            text = status.text
            name = status.user.screen_name
            id_str = status.id_str
            created = status.created_at
            entities = str(status.entities)
            ext_entities = {}

            table = db["tweets"]
            try:
                table.insert(dict(
                    Text=text,
                    Author=name,
                    Id_str=id_str,
                    Date=created,
                    Entities = entities,
                    Extended_Entities = str(ext_entities),
                ))
            except ProgrammingError as err:
                print(err)
            
            
 

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True  # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True  # Don't kill the stream



auth = tweepy.OAuthHandler(settings.consumer_key, settings.consumer_secret)
auth.set_access_token(settings.access_token, settings.access_token_secret)
api = tweepy.API(auth,timeout=200)

stream_listener = CustomStreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=settings.TRACK_TERMS,)