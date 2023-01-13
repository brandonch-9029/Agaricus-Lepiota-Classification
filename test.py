### IMPORTS AND READ CSV

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Initialise dataframe with headers
names = ["lettr", "x-box", "y-box", "width", "high ", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
letter = pd.read_csv("letter-recognition.data", names=names)

X = letter.drop(columns=["lettr"]) 
y = letter["lettr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.4)

## No preprocessing done yet but my model achieves a result of 90% with 80/20



