# AI Doctor

 This project serves as an AI Doctor, merging medical expertise with machine learning. With the use of AI image recognition, it can detect human body parts and provide possible preliminary diagnoses. 

The future is here! AI Doctor [Imgur](https://imgur.com/Nsvcgan)


## The Algorithm

This algorithm uses the detectnet, a part of NVIDIA's jetson-inference library. The data consisted of around 5,000 images of the three different body parts that are being tested, the head, leg, and arm. First, the data was trained through the detection module of the jetson-inference package. Through the training. the model identifies and analyzes the different anatomical structures. This algorithm then is able to detect the image through the webcam and identify the exact human body part associated with the image. 

## Running this project

1: Click 'New Terminal'
2: Type 'python3 finals.py /dev/video0'
3: Show the body part to the webcam

Note: Install jetson-inference package if it is not already installed

View a video explanation here: https://youtu.be/wl97W8klouA
