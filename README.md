# Facial-Expression-Recognition

## Project Overview
- The aim of project is to classify people's emotions based on their face images.
- Project is divided into two parts which is combined to give output:
 1. Facial key points detection model
 2. Facial expression detection model
- There is around 20000 facial images, with their associated facial expression lables and 2000 images with their facial key-point annotations.
- To train a model that automatically shows the people emotions and expression.

## Model 1: Key Facial Point Detection
- Created a deep learning model bases on convolutional neural network (CNN) and Residual Block to predict facial keypoints.
- Dataset contsists of x and y coordinates of 15 facial key points.
- Input images are 96x96 pixels.
- Images consits of only one color channel i.e images are grayscaled.
- Dataset Source: [Kaggle](http://https://www.kaggle.com/c/facial-keypoints-detection/data "Kaggle")

![pic1](https://user-images.githubusercontent.com/42632417/110663048-1f295700-81ec-11eb-87f8-9b424fb2141f.png)
#### 
How?
- Dataset contsists of x and y coordinates of 15 facial key points
**Input Image -> Trained Key Facial Points -> Detector Model**

![dsBuffer bmp](https://user-images.githubusercontent.com/42632417/110666470-5ea57280-81ef-11eb-8113-cc9a9690587d.png)

## Model 2: Facial Expression Detection
- This model classifies people's emotion.
- Data contains images that belongs to five categories:
-  0 : Angry 
-  1 : Disgust
-  2 : Sad
-  3 : Happy
-  4 : Surprise


![download](https://user-images.githubusercontent.com/42632417/110667195-0ae75900-81f0-11eb-835a-79a92334bf47.png)

Dataset Source: [Kaggle](http://https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data "Kaggle")

### How?

![pic](https://user-images.githubusercontent.com/42632417/110667735-91039f80-81f0-11eb-9ef4-7dc7bcbdf9a7.GIF)

- RESNET(RESIDUAL NETWORK) as deep learning model
Resnet includes skip connections feautres which enables training of 152 layers without vanishing gradient issue.


## Methodology Flowchart

The figure below shows flowchart of our proposed methedology:

![pic](https://user-images.githubusercontent.com/42632417/110668836-bba22800-81f1-11eb-8469-99c64409f098.GIF)



## Classification Report
|	| precision |    recall | f1-score  | support |
|:--:	| :---:     |   :---:   | :---:	    | :---:   |
|   0   |   0.78    |    0.76   |   0.77    |   249   | 
|   1   |   1.00    |    0.73   |   0.84    |    26   |
|   2   |   0.79    |    0.83   |   0.81    |   312   |
|   3   |   0.92    |    0.94   |   0.93    |   434   |
|   4   |   0.96    |    0.88   |   0.91    |   208   | 
| accuracy|         |           |   0.86    |  1229   | 
|macro avg| 0.89    |  0.83     | 0.85      | 1229    |
|weighted avg|0.86   |   0.86   |   0.86    | 1229    |

## Performance of the model
- with 500 epochs
![download](https://user-images.githubusercontent.com/42632417/110739242-c93dc900-8256-11eb-9218-2de1909aaa25.png) ![download](https://user-images.githubusercontent.com/42632417/110739261-d1960400-8256-11eb-8744-32c6e4862207.png)

## Final Result

![download](https://user-images.githubusercontent.com/42632417/110671267-5439a780-81f4-11eb-9725-3e42d60e094d.png)

### Dataset full : https://drive.google.com/drive/folders/1lgB9ZouVrk9xuyiJ-dp4yR4GRXhrgJiS?usp=sharing

## How to run?
- Download dataset from above Gdrive link
- clone repo
- run Facial Expression Recognition.ipynb on colab/notebook.

# References:
- https://www.youtube.com/channel/UC76VWNgXnU6z0RSPGwSkNIg
