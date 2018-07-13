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
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from Twitter.twitter_functions import *
from Twitter import private


# read in our csv from the twitter connection

tweets = check_for_hist("../Twitter/")

col = ['Hashtags','urls','media_url']

column_creator(tweets,col)

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


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


PATH_TO_TEST_IMAGES_DIR = '../images/' # set path of where photos were downloaded to

photos = os.listdir(path=PATH_TO_TEST_IMAGES_DIR) # create list of files in folder

# create
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i[:-4])) for i in photos if i[-4:] == '.jpg']

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 12)


to_tweet = {'to_tweet':[],
            'to_tweet_file':[],
            'image_score':[],
           }

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
            vis_util.visualize_boxes_and_labels_on_image_array(image=image_np,
                                                               boxes = output_dict['detection_boxes'],
                                                               classes = output_dict['detection_classes'],
                                                               scores = output_dict['detection_scores'],
                                                               category_index = category_index,
                                                               instance_masks = output_dict.get('detection_masks'),
                                                               min_score_thresh = threshold,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)
            for index, value in enumerate(classes[0]):
                if scores[0, index] > threshold and (category_index.get(value)).get('name') == 'huracan':
                    to_tweet['to_tweet'].append(1)
                    to_tweet['to_tweet_file'].append(image_path)
                    to_tweet['image_score'].append(scores[0][0])
                    # plt.figure(figsize=IMAGE_SIZE)
                    # plt.imshow(image_np)


to_tweet = pd.DataFrame(data=to_tweet)

pattern = '.+\/(\w.+)'
try:
    to_tweet['to_tweet_file'] = to_tweet['to_tweet_file'].str.extract(pattern)
except:
    pass

tweets['post'] = tweets['image_name'].isin(to_tweet['to_tweet_file'])

auth = tweepy.OAuthHandler(private.consumer_key, private.consumer_secret)
auth.set_access_token(private.access_token, private.access_token_secret)

api = tweepy.API(auth)
status = "beep boop I'm an Image recognition Bot #Huracan #Lamborghini "

for index,image in tweets.iterrows():
    if image['post'] == True:
        status = "beep boop I'm an Image recognition Bot #Huracan #Lamborghini " + image['urls']
        tweet_it = api.update_with_media(filename="../images/"+str(image['image_name']),status= status)

if 'already_seen.csv' not in os.listdir("../Twitter/"):
    print('creating new file')
    tweets.to_csv("../Twitter/already_seen.csv")
else:
    print('appending to csv')
    with open("../Twitter/already_seen.csv", 'a') as f:
        tweets.to_csv(f, mode='a', header=False)

for file in os.listdir("../images/"):
        if file[-4:] == '.jpg':
            os.remove("../images/" + file)