#! /Users/brianosgood/.virtualenvs/huracan/bin/python
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tweepy
# import matplotlib.pyplot as plt
from PIL import Image
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
import subprocess
import gspread_dataframe as gd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from Twitter.twitter_functions import *
from Twitter import private



# read in our csv from the twitter connection

tweets = check_for_hist("../Twitter/") # check_for_hist function looks to see if there are any tweets that have already been seen

col = ['Hashtags','urls','media_url_https','external_url'] # columns to create through column_creator function

column_creator(tweets,col) # creating columns

# Download the photos into a folder for processing
get_media(tweets)


# What model to load
MODEL_NAME = '../object_detection/lambo_detection_graph/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../object_detection/data', 'lambo_detection.pbtxt')

# number of classes in pbtxt file
NUM_CLASSES = 4
 
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
to_tweet = {'to_tweet':[],
            'to_tweet_file':[],
            'image_score':[],
            'car_type':[],
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
            threshold = 0.8

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
                    to_tweet['to_tweet'].append(1)
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('huracan')
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
                elif scores[0, index] > 0.8 and (category_index.get(value)).get('name') == 'aventador':
                    to_tweet['to_tweet'].append(1)
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('aventador')
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
                elif scores[0, index] > 0.98 and (category_index.get(value)).get('name') == 'gallardo':
                    to_tweet['to_tweet'].append(1)
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('gallardo')
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
                elif scores[0, index] > 0.98 and (category_index.get(value)).get('name') == 'murcielago':
                    to_tweet['to_tweet'].append(1)
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    to_tweet['car_type'].append('murcielago')
#                     plt.figure(figsize=IMAGE_SIZE)
#                     plt.imshow(image_np)
                else:
                    pass

# create dataframe from dictionary
to_tweet = pd.DataFrame(data=to_tweet)

# # extract file name from file url to compare if file is in main tweets dataframe
pattern = '.+\/(\w.+)'
try:
    to_tweet['image_name'] = to_tweet['to_tweet_file'].str.extract(pattern)
except:
    to_tweet['image_name'] = ""

# checking to see if to_tweet_file is in tweets['image_name'] if it is, then set post column value to True at that spot
tweets['post'] = tweets['image_name'].isin(to_tweet['image_name'])
tweets['car_type'] = ""
tweets['image_score'] = ""
for i,x in tweets.iterrows():
    for j,y in to_tweet.iterrows():
        if y['image_name'] == x['image_name']:
            tweets.at[i,'car_type'] = y['car_type']
            tweets.at[i,'image_score'] = y['image_score']


# Tweepy code block, instantiates API connection, writes status and posts photo if photo was marked True
auth = tweepy.OAuthHandler(private.consumer_key, private.consumer_secret)
auth.set_access_token(private.access_token, private.access_token_secret)

api = tweepy.API(auth)
status = "beep boop I'm an Image recognition Bot #Huracan #Lamborghini "

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
    else:
        pass

        
# checking if this is a first run, if it is, create csv called 'already_seen'
if 'already_seen.csv' not in os.listdir("../Twitter/"):
    print('creating new file')
    tweets.to_csv("../Twitter/already_seen.csv",)
    

    
else: # if not first run, then append to csv 'already_seen'
    print('appending to csv')
    with open("../Twitter/already_seen.csv", 'a') as f:
        tweets.to_csv(f, mode='a', header=False,)

# pygspread connection:
# connection to google sheets, drop Entities and Extended Entities, convert Dates to string and append columns to google sheet
tweets.drop(labels = ['Entities'],axis=1,inplace=True)

tweets.drop(labels=['Extended Entities'],axis=1,inplace=True)

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
for file in os.listdir("../images/"):
        if file[-4:] == '.jpg':
            os.remove("../images/" + file)