
from flask import Flask, render_template, url_for, request, jsonify
import json
import joblib
import numpy as np
import spacy
from sympy import Id, Idx
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
import nltk as nltk
#nltk.download('all')
from nltk.corpus import stopwords
stopwords_En = nltk.corpus.stopwords.words('english')
from nltk.tokenize import word_tokenize
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()
from bs4 import BeautifulSoup
import string
import re
import contractions
import logging
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import requests
from flask_mysqldb import MySQL

server = Flask(__name__)
try: 
        #connection database
    server.config['MYSQL_HOST'] = 'localhost'
    server.config['MYSQL_USER'] = 'root'
    server.config['MYSQL_PASSWORD'] = ''
    server.config['MYSQL_DB'] = 'cnn'
    print("connected")
except:
    print ("I am unable to connect to the database")


mysql = MySQL(server)


  #function scarp data from cnn 
def file_article(cnn):
    
    CNN_article = {'author': '', 'date_published': '','part_of': '','article_section': '','url': '','headline': '','description': '','keywords': '','alternative_headline': '','text': ''}
                                 
    #is_article = True
    response = requests.get(cnn)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        CNN_article['author'] = soup.find(itemprop="author").attrs['content']
        CNN_article['date_published'] = soup.find(itemprop="datePublished").attrs['content']
        CNN_article['part_of'] = soup.find(itemprop="isPartOf").attrs['content']
        CNN_article['article_section'] = soup.find(itemprop="articleSection").attrs['content']
        CNN_article['url'] = soup.find(itemprop="url").attrs['content']
        CNN_article['headline'] = soup.find(itemprop="headline").attrs['content']
        CNN_article['description'] = soup.find(itemprop="description").attrs['content']
        CNN_article['keywords'] = soup.find(itemprop="keywords").attrs['content']
        CNN_article['alternative_headline'] = soup.find(itemprop="alternativeHeadline").attrs['content']
        CNN_article['text'] = soup.find(id="body-text").text

    except Exception as err:
        #is_article = False
        print(f'Missing Article dataError: {err}')

    return(CNN_article)

 # reprocess data from scrap 
def information_interval(text):
    text=contractions_fix0(text)
    text=remove_punct(text)
    text=remove_Space(text)
    text=remove_lower(text)
    text=lemm(text)
    text=tokenize(text)
    text=remove_stopwords(text)
    return(text)
def contractions_fix0(text):
    text=contractions.fix(str(text), slang=True)
    return(text)
def remove_punct(text):
    text=re.sub(r"[^a-zA-Z0-9]", " ", str(text))
    return str(text)
def remove_Space(text):
    text=re.sub(' +',' ',str(text))
    return str(text)   
def remove_lower(text):
    text=text.lower()
    return str(text) 

def lemm(x):
  x=str(x)
  x_list=[]
  doc=sp(x)
  for token in doc:
    lemma=token.lemma_
    if lemma == "-PRON-" or lemma=='be':
       lemma=token.text

    x_list.append(lemma)
  return ' '.join(x_list)

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens
def remove_stopwords(tokenized_list):
    text = " ".join([word for word in tokenized_list  if len(word) > 2 if word not in all_stopwords])
    return text

   
@server.route('/result',methods=['POST','GET'])
def result():
    model_flag=False
    # take url from user
    cnn=request.form.to_dict()
    artical_cnn=cnn['name']
    # pass to scarp function
    artical_Cnn=file_article(artical_cnn)
    #print(artical_Cnn['text'])
    #pass to reprocess function 
    text0=information_interval(artical_Cnn['text'])
    # which model pass to data
    if artical_Cnn['part_of'] =='news':
            if artical_Cnn['article_section'] =='us':
                    tfidf=joblib.load('tfidf_df_us.pkl')
                    model=joblib.load('df_us _model.pkl')
            elif artical_Cnn['article_section'] =='uk': 
                    tfidf=joblib.load('tfidf_df_uk.pkl')
                    model=joblib.load('df_uk_model.pkl')
            elif artical_Cnn['article_section'] =='opinions':
                    tfidf=joblib.load('tfidf_df_opinions.pkl')
                    model=joblib.load('df_opinions_model.pkl')      
            elif artical_Cnn['article_section'] =='world':
                    tfidf=joblib.load('tfidf_df_world.pkl')
                    model=joblib.load('df_world_model.pkl')
            elif artical_Cnn['article_section'] =='australia':
                    tfidf=joblib.load('tfidf_df_australia.pkl')
                    model=joblib.load('df_australia_model.pkl')
            elif artical_Cnn['article_section'] =='asia':
                    tfidf=joblib.load('tfidf_df_asia.pkl')
                    model=joblib.load('df_asia_model.pkl')  
            elif artical_Cnn['article_section'] =='americas':
                    tfidf=joblib.load('tfidf_df_americas.pkl')
                    model=joblib.load('df_americas_model.pkl')
            elif artical_Cnn['article_section'] =='opinion':
                    tfidf=joblib.load('tfidf_df_opinion.pkl')
                    model=joblib.load('df_opinion_model.pkl')
            elif artical_Cnn['article_section'] =='africa':
                    tfidf=joblib.load('tfidf_df_africa.pkl')
                    model=joblib.load('df_africa_model.pkl')  
            elif artical_Cnn['article_section'] =='middleeast':
                    tfidf=joblib.load('tfidf_df_middleeast.pkl')
                    model=joblib.load('df_middleeast_model.pkl')
            elif artical_Cnn['article_section'] =='weather':
                    tfidf=joblib.load('tfidf_df_weather.pkl')
                    model=joblib.load('df_weather_model.pkl')
            elif artical_Cnn['article_section'] =='china':
                    tfidf=joblib.load('tfidf_df_china.pkl')
                    model=joblib.load('df_china_model.pkl')  
            elif artical_Cnn['article_section'] =='living':
                    tfidf=joblib.load('tfidf_df_living.pkl')
                    model=joblib.load('df_living_model.pkl')
            elif artical_Cnn['article_section'] =='india':
                    tfidf=joblib.load('tfidf_df_india.pkl')
                    model=joblib.load('df_india_model.pkl')
            elif artical_Cnn['article_section'] =='justice':
                    tfidf=joblib.load('tfidf_df_justice.pkl')
                    model=joblib.load('df_justice_model.pkl')            
            else:
                 model_flag=True                
    elif artical_Cnn['part_of'] =='sport':
            if artical_Cnn['article_section'] =='sports':
                    tfidf=joblib.load('tfidf_df_sports.pkl')
                    model=joblib.load('df_sports_model.pkl')
            elif artical_Cnn['article_section'] =='football':
                    tfidf=joblib.load('tfidf_df_football.pkl')
                    model=joblib.load('df_football_model.pkl')  
            elif artical_Cnn['article_section'] =='tennis':
                    tfidf=joblib.load('tfidf_df_tennis.pkl')
                    model=joblib.load('df_tennis_model.pkl')
            elif artical_Cnn['article_section'] =='golf':
                    tfidf=joblib.load('tfidf_df_golf.pkl')
                    model=joblib.load('df_golf_model.pkl')
            elif artical_Cnn['article_section'] =='motorsport':
                    tfidf=joblib.load('tfidf_df_motorsport.pkl')
                    model=joblib.load('df_motorsport_model.pkl')            
            else:
                 model_flag=True                
    elif artical_Cnn['part_of'] =='travel':
            tfidf=joblib.load('tfidf_df_travel.pkl')
            model=joblib.load('df_travel_model.pkl') 
   
    elif artical_Cnn['part_of'] =='health':
            tfidf=joblib.load('tfidf_df_health.pkl')
            model=joblib.load('df_health_model.pkl') 
    elif artical_Cnn['part_of'] =='politics':
            tfidf=joblib.load('tfidf_df_politics.pkl')
            model=joblib.load('df_politics_model.pkl') 
    elif artical_Cnn['part_of'] =='business':
            tfidf=joblib.load('tfidf_df_business.pkl')
            model=joblib.load('df_business_model.pkl') 
    else:
             model_flag=True     
    
    
    
    print(text0)
    if model_flag==False:
            # transfrom text 
            b=tfidf.transform([text0])
            b= np.array(b.toarray())
            # pass to model return index of artical
            distance,idx = model.kneighbors(b)
            print(idx)
            idx=idx[0]
            idx=idx.tolist()
            id_list = idx
            #convert list  index to tuple 
            id_tuple = tuple(id_list)
            print(id_tuple[:-1])
            cursor = mysql.connection.cursor()
            # pass tuple to select qureey to fitch data from database
            cursor.execute("SELECT headline,text FROM df_business WHERE id IN {};".format(id_tuple))
            articals_interval_Data=cursor.fetchall()
            print(articals_interval_Data[:-1])
    else:
        articals_interval_Data=1
    return render_template('index.html',articals_interval_Data=articals_interval_Data)
@server.route('/',methods=['GET'])
@server.route('/home',methods=['GET'])
def home():
   return render_template('index.html')
server.run()
