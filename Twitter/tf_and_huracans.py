
import os

#this file runs the entire PyFilter Project using subprocess commands to run the files
import time
starttime=time.time()
#detect = "sudo jupyter nbconvert --execute ../object_detection/lambo_detection.ipynb"
detect = "sudo python3 ../object_detection/twitter_detection.py"
while True:
    print('Running Detection')
    os.system(detect)
    print("Processed")
    time.sleep(120.0 - ((time.time() - starttime) % 120.0))
