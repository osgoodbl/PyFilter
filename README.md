# PyFilter

![Object Detection](https://pbs.twimg.com/profile_banners/1016789079566921728/1531262895/1500x500 "Object Detection")[@tf_and_lambos](https://twitter.com/tf_and_lambos)

The inspiration for this project was actually from my instagram as I follow #Lamborghini, #Huracán and a couple other hashtags. I noticed that a lot of the photos tagged with those hashtags didn't actually feature the item I wanted to see. So I thought maybe I could make an image object detector to filter posts on social media that incorrectly use hashtags on a photo to gain a wider audience even if the photo is unrelated to the hashtag.

# Project Files

If you want to build you own object detector with TensorFlow follow the Object_Detection_Part# series.
If you want to see how Tweepy can follow Tweets, and how I loaded the resulting data into an object detection, check out the Twitter directory.

# Object Detection Setup

* Step one: use a Python3 environment, this project was setup on a [DigitalOcean](https://m.do.co/c/ad1f572a2ff5) machine learning droplet.
* Step two: open Object_Detection_Part1 and work through all the notebooks
* Step three: crack a beer as the model trains
* step four: see how your model did!


# TensorFlow Object Detection and Tweepy

`pip install tweepy`

## Tweepy
This package was tricky, the documentation is spread out over stackoverflow and
Twitter's API page. My initial approach was to create a RESTapi query which,
worked perfectly, however the entire system had to be run individually and this
would eat up a lot of computation power as it would constantly create and remove
csv files.
Enter the Twitter Stream portion of Tweepy, this allows for a real time monitoring
of the Twitter-verse on specific hashtags (perfect!).

Next steps for Tweepy streaming setup was to collect the correct information from
Twitter. The data is received as a JSON, but after writing that information to a CSV,
the data is converted to strings. This led to a problem with parsing out hashtags,
media urls, and any other entities. My solution, was the utilize the "converters" argument in `pandas.read_csv()` along with a custom Jsonparser function.

The initial format of the table:

| Author |  Id | Date | Text | Entities | Extended Entities |

|@Twitter | ### | 7/13/2018 | json as text | json as text |      

Using the JsonParser function I was able to transform the json from text into actual JSON format.
```
def JsonParser(data):
    '''parser to parse JSON test correctly from loaded csv'''
    import json
    if data == "":
        pass
    else:
        data = json.dumps(ast.literal_eval(data))
        j1 = json.loads(data)
    return j1
```


This enabled me to properly call out sub dictionaries in a column and create new columns with the values I wanted from the JSON.

Once I had the csv loaded in and cleaned, I was able to download the photos in each post and pass them into my trained model with a pseudo pipeline code structure.

## TensorFlow

The next challenge was processing the downloaded photos and returning only the photos that are above a certain confidence value. Using ssd_mobilenet_v1_coco_2018_01_28 with TensorFlow, the images are processed quickly. Post processing generates dictionaries of arrays with an individual score for each pixel and then returns the whole score for that image. Parsing out the class, score and image name allowed for me to generate a dataframe that can be cross referenced with the Twitter data to mark a Tweet at "okay to post" or "ignore".

## Posting with Tweepy

Creating a Tweet with Tweepy was very straight forward and has lots of options. I chose the update_status_with_media to post a photo along with a comment.

# Moving to production

As I stated at the beginning, I used DigitalOcean. The "droplet" or instance on their server runs Ubuntu, without my virtual environment. Using `pip freeze` I could generate a requirements.txt file to tell the droplet what packages to install. However, there are still some system level packages that are not pip installed directly.

This proved challenging and was remedied after many `pip3 installs`.

# The future of PyFilter

* I plan on expanding out to the rest of the Lamborghini family of vehicles
* Metrics on the specific tweets I pull in (who gets mentioned the most, other hashtags used with the post)
* Sentiment analysis
* Instagram connection
# Resources:

slides:['PyFilter Presentation'](https://docs.google.com/presentation/d/18qpbjyiS_k5-6ccuQr7dnobMsWEgFSdQf-_H4qA07Cs/edit?usp=sharing)

data dashboard: [DataStudio](https://datastudio.google.com/open/1pLVzUKUWiWU2NKn15vG9SoWRnf6-ERop)

guidance:
* [TensorFlow](https://www.tensorflow.org/)
* [Python Programming](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/)

* [COCO Dataset Model]
