from flask import Flask, render_template, request
import pandas as pd
from joblib import load 
from urllib.request import urlopen

def count_special_characters(input_string):
    special_characters = 0
    for char in input_string:
        if not char.isalnum() and not char.isspace():
            special_characters += 1
    return special_characters

model=load('rf_model.joblib')
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
        print(url)
        pred=model.predict([url])[0]
        print('My prediction is: '+str(pred))
        
        if pred==1:
            result="https://i.ibb.co/PjrWmJm/cross.png"
        else:
            result="https://i.ibb.co/0hk28GL/check.png"

        return render_template('index.html',text=text,result=result)

if __name__ == "__main__":
    app.run(debug=True)
