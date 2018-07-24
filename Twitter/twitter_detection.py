#! /Users/brianosgood/.virtualenvs/huracan/bin/python
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tweepy
# import matplotlib.pyplot as plt
from PIL import Image
import sqlite3
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
import subprocess
import gspread_dataframe as gd
import gspread
import private
from twitter_functions import *
from oauth2client.service_account import ServiceAccountCredentials
sys.path.append("..")
from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util





tweets = check_for_hist("./") # check_for_hist function looks to see if there are any tweets that have already been seen

col = ['Hashtags','urls','media_url_https','external_url'] # columns to create through column_creator function

column_creator(tweets,col) # creating columns

# Download the photos into a folder for processing
get_media(tweets,"../images")

print('Photos Downloaded')

# What model to load
MODEL_NAME = '../models/research/object_detection/object_detection_graph_28737/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../models/research/object_detection/data_1', 'lambo_detection.pbtxt')

# number of classes in pbtxt file
NUM_CLASSES = 6
 
# instantiate Tensorflow graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# create labels mapping
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# path for downloaded images location
PATH_TO_TEST_IMAGES_DIR = '../images/' # set path of where photos were downloaded to

photos = os.listdir(path=PATH_TO_TEST_IMAGES_DIR) # create list of files in folder

# create path for each image
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i[:-4])) for i in photos if i[-4:] == '.jpg']

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 12)

# Dictionary to append photo values to 
to_tweet = {'to_tweet_file':[],
            'image_score':[],
            'car_type':[],
            'huracan':[],
            'aventador':[],
            'gallardo':[],
            'murcielago':[],
            'urus':[],
            'not a lambo':[],

           }
#running object detection and choosing to post
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        object_dict = {}
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            threshold = 0.90

            (boxes, scores, classes, num) = sess.run(
                                                    [detection_boxes,
                                                     detection_scores,
                                                     detection_classes,
                                                     num_detections],
                                                     feed_dict={image_tensor: image_np_expanded})
#'''Uncomment below in a jupyter notebook to view image bounding boxes'''
#            vis_util.visualize_boxes_and_labels_on_image_array(image=image_np,
#                                                               boxes = output_dict['detection_boxes'],
#                                                               classes = output_dict['detection_classes'],
#                                                               scores = output_dict['detection_scores'],
#                                                               category_index = category_index,
#                                                               instance_masks = output_dict.get('detection_masks'),
#                                                               min_score_thresh = threshold,
#                                                               use_normalized_coordinates=True,
#                                                               line_thickness=8)


            for index, value in enumerate(classes[0]):
                if scores[0, index] > threshold and (category_index.get(value)).get('name') == 'huracan':
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('huracan')
                    to_tweet['huracan'].append(1)
                    to_tweet['not a lambo'].append(0)
                    to_tweet['urus'].append(0)
                    to_tweet['aventador'].append(0)
                    to_tweet['gallardo'].append(0)
                    to_tweet['murcielago'].append(0)
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
#                     plt.title('huracan'+ str(scores[0][0]))
                elif scores[0, index] > threshold and (category_index.get(value)).get('name') == 'aventador':
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('aventador')
                    to_tweet['aventador'].append(1)
                    to_tweet['not a lambo'].append(0)
                    to_tweet['huracan'].append(0)
                    to_tweet['urus'].append(0)
                    to_tweet['gallardo'].append(0)
                    to_tweet['murcielago'].append(0)
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
#                     plt.title('aventador'+ str(scores[0][0]))
                elif scores[0, index] > threshold and (category_index.get(value)).get('name') == 'gallardo':
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('gallardo')
                    to_tweet['gallardo'].append(1)
                    to_tweet['not a lambo'].append(0)
                    to_tweet['huracan'].append(0)
                    to_tweet['aventador'].append(0)
                    to_tweet['urus'].append(0)
                    to_tweet['murcielago'].append(0)
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
#                     plt.title('gallardo'+ str(scores[0][0]))
                elif scores[0, index] > threshold and (category_index.get(value)).get('name') == 'murcielago':
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('murcielago')
                    to_tweet['murcielago'].append(1)
                    to_tweet['not a lambo'].append(0)
                    to_tweet['huracan'].append(0)
                    to_tweet['aventador'].append(0)
                    to_tweet['gallardo'].append(0)
                    to_tweet['urus'].append(0)
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
#                     plt.title('Murci'+ str(scores[0][0]))
                elif scores[0, index] > threshold and (category_index.get(value)).get('name') == 'not a lambo':
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('not a lambo')
                    to_tweet['not a lambo'].append(1)
                    to_tweet['urus'].append(0)
                    to_tweet['huracan'].append(0)
                    to_tweet['aventador'].append(0)
                    to_tweet['gallardo'].append(0)
                    to_tweet['murcielago'].append(0)
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
#                     plt.title('NOPE'+ str(scores[0][0]))
                elif scores[0, index] > threshold and (category_index.get(value)).get('name') == 'urus':
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('urus')
                    to_tweet['urus'].append(1)
                    to_tweet['not a lambo'].append(0)
                    to_tweet['huracan'].append(0)
                    to_tweet['aventador'].append(0)
                    to_tweet['gallardo'].append(0)
                    to_tweet['murcielago'].append(0)
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
#                     plt.title('urus'+ str(scores[0][0]))
                else:
                    pass
                    
        
# create dataframe from dictionary
to_tweet = pd.DataFrame(data=to_tweet)

# extract file name from file url to compare if file is in main tweets dataframe
pattern = '.+\/(\w.+)'
try:
    to_tweet['image_name'] = to_tweet['to_tweet_file'].str.extract(pattern)
except:
    to_tweet['image_name'] = ""
tweets['car_type'] = ""
tweets['car_type'] = ""
tweets['urus'] = ""
tweets['not a lambo'] = ""
tweets['huracan'] = ""
tweets['aventador'] = ""
tweets['gallardo'] = ""
tweets['murcielago'] = ""
# checking to see if to_tweet_file is in tweets['image_name'] if it is, then set post column value to True at that spot
for i,x in tweets.iterrows():
    for j,y in to_tweet.iterrows():
        if y['image_name'] == x['image_name']:
            tweets.at[i,'car_type'] = y['car_type']
            tweets.at[i,'image_score'] = y['image_score']
            tweets.at[i,'huracan'] = y['huracan']
            tweets.at[i,'aventador'] = y['aventador']
            tweets.at[i,'gallardo'] = y['gallardo']
            tweets.at[i,'murcielago'] = y['murcielago']
            tweets.at[i,'urus'] = y['urus']
            tweets.at[i,'not a lambo'] = y['not a lambo']


# Tweepy code block, instantiates API connection, writes status and posts photo if photo was marked True
auth = tweepy.OAuthHandler(private.consumer_key, private.consumer_secret)
auth.set_access_token(private.access_token, private.access_token_secret)

api = tweepy.API(auth)

for index,image in tweets.iterrows():
    if image['car_type'] == 'huracan':
        status = ( "With a {}% Confidence, I think there is a Huracan in this photo #Huracan #Lamborghini ".format(round(100*image['image_score'],3)) + image['urls'])
#         print(status)
        tweet_it = api.update_with_media(filename="../images/"+str(image['image_name']),status= status)
    elif image['car_type'] == 'aventador':
        status = ( "With a {}% Confidence, I think there is a Aventador in this photo #Aventador #Lamborghini ".format(round(100*image['image_score'],3)) + image['urls'])
#         print(status)
        tweet_it = api.update_with_media(filename="../images/"+str(image['image_name']),status= status)
    elif image['car_type'] == 'gallardo':
        status = ( "With a {}% Confidence, I think there is a Gallardo in this photo #Gallardo #Lamborghini ".format(round(100*image['image_score'],3)) + image['urls'])
#         print(status)
        tweet_it = api.update_with_media(filename="../images/"+str(image['image_name']),status= status)
    elif image['car_type'] == 'murcielago':
        status = ( "With a {}% Confidence, I think there is a Murcielago in this photo #Murcielago #Lamborghini  ".format(round(100*image['image_score'],3)) + image['urls'])
#         print(status)
        tweet_it = api.update_with_media(filename="../images/"+str(image['image_name']),status= status)
    elif image['car_type'] == 'urus':
        status = ( "With a {}% Confidence, I think there is a Urus in this photo #Murcielago #Lamborghini  ".format(round(100*image['image_score'],3)) + image['urls'])
#         print(status)
        tweet_it = api.update_with_media(filename="../images/"+str(image['image_name']),status= status)
    else:
        pass

        
print('appending to database')
tweets = tweets.applymap(str)
tweets.to_sql('hist_tweets',sqlite3.connect("../Twitter/hist_tweets.db"),if_exists='append',index=True, index_label='id')
print('appended')

# pygspread connection:
# connection to google sheets, drop Entities and Extended Entities, convert Dates to string and append columns to google sheet
tweets.drop(labels = ['Entities'],axis=1,inplace=True)

tweets.drop(labels=['Extended_Entities'],axis=1,inplace=True)

tweets['Date'] = tweets['Date'].map(lambda x: tz_convert(x))
tweets['Date'] = tweets['Date'].astype(str) #write PST datetime to string so it can be appended to Google Sheets

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('./PyFilter-34d3cda723bf.json',scope)

gc = gspread.authorize(credentials)

ws = gc.open("PyFilter").worksheet("Twitter_Data") #open google sheet and worksheet
existing = gd.get_as_dataframe(worksheet=ws) #get worksheet as dataframe
updated = existing.append(tweets, ignore_index=False,sort=False)

gd.set_with_dataframe(ws,updated,resize=True)
print('appended to google sheet')


# delete photos that have been downloaded
for file in os.listdir("../images"):
        if file[-4:] == '.jpg':
            os.remove("../images/" + file)
            
print('Photos Removed from Folder')
print('Executed Successfully')