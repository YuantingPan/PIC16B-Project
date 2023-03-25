# PIC16B-Project

In this project, we aim to develop a machine learning model to detect and analyze humans’ facial expressions. The model will receive a picture of a human’s face, and gives a prediction of facial expressions. In addition, we developed a webapp, wehre users can upload their face images to get facial expression analysis.

Collaborators: Euibin Kim, Lei Xu, Yuanting Pan

We obtained our dataset from [www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset](www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

Copy of the latest training on Colab: [here](https://colab.research.google.com/drive/1hPL5aEelaZ2Ejdp5q9PIeSnFan5EosCB?usp=sharing)

## Project

In `Project` folder, you can find the main body of our machine learning project. This includes the dataset, the defined functions, and the training notebook, and the saved models. Use `project.ipynb` to run the model from beginning.

## Web

In `web` folder, you can find the codes in **flask** for a website integrated with facial expression detection. The user can upload a human face image to the website, and a quick analysis for facial expression will be shown.
