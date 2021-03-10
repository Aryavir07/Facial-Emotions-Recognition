# Facial-Expression-Recognition

## Project Overview
- The aim of project is to classify people's emotions based on their face images.
- Project is divided into two parts which is combined to give output:
				- 1. Facial key points detection model
				- 2. Facial expression detection model

------------
## Model 1: Key Facial Point Detection
- Dataset Source: [Kaggle](http://https://www.kaggle.com/c/facial-keypoints-detection/data "Kaggle")

![pic1](https://user-images.githubusercontent.com/42632417/110663048-1f295700-81ec-11eb-87f8-9b424fb2141f.png)
#### 
How?
- Dataset contsists of x and y coordinates of 15 facial key points
**Input Image -> Trained Key Facial Points -> Detector Model**

![dsBuffer bmp](https://user-images.githubusercontent.com/42632417/110666470-5ea57280-81ef-11eb-8113-cc9a9690587d.png)
------------


## Model 2: Facial Expression Detection
- This model classifies people's emotion.
- Data contains images that belongs to five categories:
	- **0 -> Angry**
	- **1 -> Disgust**
	- **2 -> Sad**
	- **3 -> Happy**
	- **4 -> Surprise**


![download](https://user-images.githubusercontent.com/42632417/110667195-0ae75900-81f0-11eb-835a-79a92334bf47.png)

Dataset Source: [Kaggle](http://https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data "Kaggle")

### How?
- **Input Image (48x48) -> Classifier -> Target Classes**
image goes here
------------






