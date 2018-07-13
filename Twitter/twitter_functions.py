#! /Users/brianosgood/.virtualenvs/huracan/bin/python

def JsonParser(data):
    '''parser to parse JSON test correctly from loaded csv'''
    import json
    import ast
    if data == "":
        pass
    else:
        data = json.dumps(ast.literal_eval(data))
        j1 = json.loads(data)
    return j1

def column_creator(tweets,col):
    for _ in col:
        tweets[_] = ""
        if _ == 'Hashtags':
            for i,v in tweets.iterrows():
                hashes = []
                for tag in v['Entities'].get('hashtags'):
                    hashes.append(tag['text'])
                    tweets.at[i,_] = hashes
                if 'hashtags' not in v['Entities']:
                    hashes.append('No ' + _ )
                tweets.at[i,_] = hashes
        elif _ == 'urls':
            for i,v in tweets.iterrows():
                for tag in v['Entities'].get(_):
                    tweets.at[i,'urls'] = tag['url']
                if _ not in v['Entities']:
                    tweets.at[i,'urls'] = 'No '+ _
        elif _ == 'media_url':
            for i,v in tweets.iterrows():
                if 'media'in v['Entities'].keys():
                    tweets.at[i,_] = (v['Entities']['media'][0][_])
                elif 'entities' in v['Extended Entities'].keys():
                    if 'media' in v['Extended Entities']['entities'].keys():
                        tweets.at[i,_] = (v['Extended Entities']['entities']['media'][0][_])
                    elif 'media' not in v['Extended Entities']['entities'].keys():
                        tweets.at[i,_] = 'No ' + _
                elif 'media' not in v['Entities'].keys():
                    tweets.at[i,_] = 'No ' + _
    pattern = '.+\/(\w.+)'
    tweets['image_name'] = ''
    try:
        tweets['image_name'] = tweets['media_url'].str.extract(pattern)
    except:
        pass


def check_for_hist(path):
    import os
    import pandas as pd
    if 'already_seen.csv' not in os.listdir(path):
        tweets = pd.read_csv("../Twitter/huracan.csv",converters={"Entities":JsonParser,"Extended Entities":JsonParser})
        return tweets
    else:
        already_seen = pd.read_csv("../Twitter/already_seen.csv", usecols=['Author', 'Id', 'Date', 'Text', 'Entities', 'Extended Entities'])
        new_tweets = pd.read_csv("../Twitter/huracan.csv") #read in our csv from the twitter connection
        tweets = duplicate_drop(already_seen,new_tweets).reset_index(drop=True)
        return tweets


def get_media(tweets):
    import wget
    for url in tweets['media_url']:
        try:
            if ".jpg" in url:
                wget.download(url, out=f'../images/')
        except:
            return f"something went wrong with {url}"



def load_image_into_numpy_array(image):
    import numpy as np
    '''function to load image into a numpy array'''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def duplicate_drop(df1,df2):
    import pandas as pd
    tweets = pd.concat([df1,df2])
    tweets = tweets.drop_duplicates(keep=False)
    tweets.reset_index(drop=True)
    tweets['Entities'] = tweets['Entities'].map(lambda x: JsonParser(x))
    tweets['Extended Entities'] = tweets['Extended Entities'].map(lambda x: JsonParser(x))
    col = ['Hashtags','urls','media_url']
    column_creator(tweets,col)
    return tweets



def run_inference_for_single_image(image, graph):
  import numpy as np
  import tensorflow as tf
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks']:

        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict



def twitter_poster(tweets):
    import Twitter.private
    import tweepy
    auth = tweepy.OAuthHandler(private.consumer_key, private.consumer_secret)
    auth.set_access_token(private.access_token, private.access_token_secret)

    api = tweepy.API(auth)
    status = "beep boop I'm an Image recognition Bot #Huracan #Lamborghini "
    for index, image in tweets.iterrows():
        if image['post'] == True:
            status = "beep boop I'm an Image recognition Bot #Huracan #Lamborghini "
            api.update_with_media(filename="./test_images/" + str(image['image_name']), status=status)
    return "Tweets posted"
