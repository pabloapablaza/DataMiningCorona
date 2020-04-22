#Importar librerias matematicas
import io
import random
import string
import warnings
import pandas as pd
from pandas import ExcelWriter
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

#Importar natural languaje toolkit
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

# Importar sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#Importaciones de python
import re
import json
from collections import Counter
import glob
import os
from glob import iglob

#Herramientas de visualizacion
from matplotlib import pyplot as pyplot
import seaborn as sns 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from wordcloud import WordCloud
from tqdm import tqdm_notebook
import pickle

###############################################
###############################################

#Visualizacion de tweets
path_df = r"D:\data_mining\coronavirus-covid19-tweets"                  #Directorio del los .csv
tweets_df = pd.concat(map(pd.read_csv, glob.glob(path_df+'/*.csv')))    #Crear el dataframe con todos los archivos
print(tweets_df.head())
print(len(tweets_df))

#Guardar en CSV
def save_csv():
    print("******Inicio guardado en CSV******")
    path_write="D:/data_mining/saves_folder"
    tweets_df.to_csv(path_write+'tweets_corona.csv')
    print("******Fin guardado en CSV******")