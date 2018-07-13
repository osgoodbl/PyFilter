# PyFilter

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">beep boop I&#39;m an Image recognition Bot <a href="https://twitter.com/hashtag/Huracan?src=hash&amp;ref_src=twsrc%5Etfw">#Huracan</a> <a href="https://twitter.com/hashtag/Lamborghini?src=hash&amp;ref_src=twsrc%5Etfw">#Lamborghini</a> <a href="https://t.co/ZQVQwPcpUB">pic.twitter.com/ZQVQwPcpUB</a></p>&mdash; TensorFlow and Huracáns (@tf_and_huracans) <a href="https://twitter.com/tf_and_huracans/status/1016917233275031553?ref_src=twsrc%5Etfw">July 11, 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


![Object Detection](https://pbs.twimg.com/profile_banners/1016789079566921728/1531262895/1500x500 "Object Detection")[@tf_and_huracans](https://twitter.com/tf_and_huracans)

The inspiration for this project was actually from my instagram as I follow #Lamborghini, #Huracán and a couple others. I noticed that a lot of the photos tagged with those hashtags didn't actually feature the item I wanted to see. So I thought maybe I could make an image object detector to filter posts on social media that incorrectly use hashtags on a photo to gain a wider audience even if the photo is unrelated to the hashtag.

# Project Files

If you want to build you own object detector with TensorFlow follow the Object_Detection_Part# series.
If you want to see how Tweepy can follow Tweets, and how I loaded the resulting data into an object detection, check out the Twitter directory.

# Object Detection Setup

* Step one: use a Python3 environment, this project was setup on a [DigitalOcean](https://m.do.co/c/ad1f572a2ff5) machine learning droplet.
* Step two: open Object_Detection_Part1 and work through all the notebooks
* Step three: crack a beer as the model trains
* step four: see how your model did!


# TensorFlow Object Detection and Tweepy

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
This enabled me to properly call out sub dictionaries in a column and create new columns with the values I wanted.

Once I had the csv loaded in and cleaned, I was able to download the photos in each post and pass them into my trained model
