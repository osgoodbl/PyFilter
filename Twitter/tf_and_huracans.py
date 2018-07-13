
import os


#this file runs the entire PyFilter Project using subprocess commands to run the files
detect = "../object_detection/twitter_detection.py"
import time
starttime=time.time()
while True:
    print('Running Detection')
    os.system(detect)
    print("Processed")
    time.sleep(30.0 - ((time.time() - starttime) % 30.0))
