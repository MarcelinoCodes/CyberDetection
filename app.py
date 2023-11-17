from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score, f1_score

def ML(u):
    df = pd.read_csv(filename, header=0)
    #filling null values
    df['content_length'] = df['content_length'].fillna(0)
    df['server'] = df['server'].fillna(0)
    #dropping columns and one hot encoding
    #Try normailzation and tokenization for URL instead of just droppping(note to self)
    df = pd.get_dummies(df, columns=['server'], drop_first=True)
    df = pd.get_dummies(df, columns=['charsets'], drop_first=True)
    #creating X and y; now train test split
    y = df['mal'] 
    X = df.drop(columns = 'mal', axis=1)
    # Creating model
    rf_model = RandomForestClassifier()
    # Fitting the model 
    rf_model.fit(X_train, y_train)
    
    # Predicting values for test dataset
    y_pred = rf_model.predict(X_test)
    
    predicted_label = rf_model.predict([u])[0]
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)
    
def count_special_characters(input_string):
    special_characters = 0
    for char in input_string:
        if not char.isalnum() and not char.isspace():
            special_characters += 1
    return special_characters

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def run_app():
    if request.method=='GET':
        return render_template('index.html')
    elif request.method=='POST':
        text = request.form['text']
        url=[100]

        url.append(len(text))
        try:
          url.append(urlopen(text).info().get('Content-Length'))
        except:
          url.append(-1)

        url.append(count_special_characters(text))

        x=[]
        y=[]

        a=0
        b=0
        try:
            x=x.append(urlopen(text).info().get('server'))
        except:
            x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            a=25

        try:
            y=y.append(urlopen(text).info().get_charsets())
        except:
            y=[0,0,0,0,0,0,0]
            b=7
        
        x_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_list=[0,0,0,0,0,0,0]
        x_list_string=['server_-1','server_ATS','server_AmazonS3','server_Apache','server_Apache/2','server_Apache/2.4.52 (Ubuntu)','server_Apache/2.4.56 (Debian)','server_Apache/2.4.6','server_GSE','server_LiteSpeed','server_MerlinCDN','server_Microsoft-IIS/8.5','server_Universe','server_cloudflare','server_ddos-guard','server_gogogadgeto-server','server_gunicorn/0.17.2','server_nginx','server_nginx/1.14.0 (Ubuntu)','server_nginx/1.14.1','server_nginx/1.18.0','server_o2switch-PowerBoost-v3','server_openresty','server_openresty/1.21.4.1','server_tsa_b']
        y_list_string=["charsets_['iso-8859-15']","charsets_['ms949']","charsets_['none']","charsets_['shift_jis']","charsets_['utf-8']","charsets_['windows-1251']",'charsets_[None]']

        if a==25:
            pass
        else:
            for i in range(len(x_list_string)):
                if x==x_list_string[i]:
                    x_list[i]=1
                    break
        
        if b==7:
            pass
        else:
            for i in range(len(y_list_string)):
                if y==y_list_string[i]:
                    y_list[i]=1
                    break

        url=url+x_list+y_list

        url=pd.Series(url).fillna(-1).tolist()
        pred=ML(url)
        if pred==1:
            result="https://i.ibb.co/PjrWmJm/cross.png"
        else:
            result="https://i.ibb.co/0hk28GL/check.png"

if __name__ == "__main__":
    app.run(debug=True)
