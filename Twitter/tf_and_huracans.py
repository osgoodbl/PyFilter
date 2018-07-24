
import os

#this file runs the entire PyFilter Project using subprocess commands to run the files
import time
starttime=time.time()
detect = "sudo python3 ./twitter_detection.py"
while True:
    print('Running Detection')
    os.system(detect)
    print("Processed")
    time.sleep(45.0 - ((time.time() - starttime) % 45.0))
