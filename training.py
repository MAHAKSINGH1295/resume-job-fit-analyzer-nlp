#teaching computer how to predict job models
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
#loading all the dataset first of all
df =pd.read_csv("data/res.csv")
#now we will be taking text and labels from resume
texts =df["text"].astype(str).tolist()
labels =df["label"].astype("str").tolist()
#for converting the texts into numbers
encoder =SentenceTransformer("all-MiniLM-L6-v2")
X =encoder.encode(texts)
#for training model
model= LogisticRegression(max_iter=2500)
model.fit(X,labels)
#in case model folder doesnt exist,we will create one
os.makedirs("model",exist_ok=True)
#for saving the trained model
joblib.dump(model,"model/jobRoleModel.pkl")
print("Model has been trained and saved succesfully!")



