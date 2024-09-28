import base64
import io
import json
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import logout
from django.contrib.auth import login, authenticate  # add to imports
from .forms import LoginForm, MyForm
from django.contrib import messages
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os
from django.views.decorators.http import require_POST
#from .forms import MonthRangeForm


def login_view(request):
    form = LoginForm()
    message = ''
    if request.method == 'POST':
        form = LoginForm(request.POST)
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home/')
        else:
            return HttpResponse("Login Failed! Invalid username or passwaord")
            #message = 'Invalid username or password.'
            #return render(request, 'home.html', {'form': form, 'message':message})
           #error_message = "Invalid username or password."
           #return render(request, 'home.html', {'error-message': error_message})
    return render(request, 'login.html', {'form': form})

def home(request): 
    return render(request, 'home.html')

def about(request): 
    return render(request, 'about.html')

def predict(request):
    return render(request, 'predict.html')

def single(request):
    return render(request, 'single.html')


def file(request):
    form = MyForm()
    if request.method == 'POST':
        form = MyForm(request.POST, request.FILES)
        #form = MonthRangeForm(request.POST)
        if form.is_valid():
            if 'button1' in request.POST:
                # process the uploaded file
                #file_name = form.cleaned_data['file_name']
                file = form.cleaned_data['file']
                file_name = file.name
                csv_filename = "default.csv"
                csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static", csv_filename)
                df = pd.read_csv(csv_path)
                #csv_path = "../static/default.csv"
                #df = pd.read_csv(csv_path)
                df['EDUCATION'].replace([0, 6], 5, inplace=True)
                df['EDUCATION'].replace(5, 4, inplace=True)
                df['MARRIAGE'].replace(0, 3, inplace=True)
                le = LabelEncoder()
                df.insert(2, 'Gender_encoded', le.fit_transform(df['SEX']))
                df.insert(3, 'education_encoded', le.fit_transform(df['EDUCATION']))
                df.insert(4, 'marriage_encoded', le.fit_transform(df['MARRIAGE']))
                df = df.drop('SEX', axis=1)
                df = df.drop('EDUCATION', axis=1)
                df = df.drop('MARRIAGE', axis=1)
                df.drop('ID',axis=1, inplace=True)
                X = df.iloc[:,:-1].values
                y = df.iloc[:,-1].values
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                scaler = RobustScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                # do something with the file
                # return render(request, 'success.html')
                d = pd.read_csv(file)
                copied_d = d.copy()
                copied_d['EDUCATION'].replace([0, 6], 5, inplace=True)
                copied_d['EDUCATION'].replace(5, 4, inplace=True)
                copied_d['MARRIAGE'].replace(0, 3, inplace=True)

                le = LabelEncoder()
                    # Encode the 'Sex' column and insert it at the 3rd position
                copied_d.insert(2, 'Gender_encoded', le.fit_transform(d['SEX']))
                    # Encode the 'Sex' column and insert it at the 3rd position
                copied_d.insert(3, 'education_encoded', le.fit_transform(d['EDUCATION']))
                copied_d.insert(4, 'marriage_encoded', le.fit_transform(d['MARRIAGE']))
                copied_d = copied_d.drop('MARRIAGE', axis=1)
                copied_d = copied_d.drop('SEX', axis=1)
                copied_d = copied_d.drop('EDUCATION', axis=1)
                copied_d.drop('ID',axis=1, inplace=True)
                #X_1 = copied_d.iloc[:,:-1].values
                #y_1 = copied_d.iloc[:,-1].values
                scaled_new_data = scaler.transform(copied_d)
                c=joblib.load('mlmodel/svc_model.joblib')
                #preds = c.predict(scaled_new_data)
                scores = c.decision_function(scaled_new_data)
                new_threshold = -0.15
            # probs = c.predict_proba(scaled_new_data)[:, 1]
        
        # apply the new threshold value
                preds = (scores >= new_threshold).astype(int)
                
                #if(preds==1):
                   # ans='The customer is probably going to DEFAULT'
                #elif(preds==0):
                    #ans = 'The custoner is a Non-Defaulter'
                #predictions = {'predictions': list(preds)}
                #return JsonResponse(predic
                #context = {'preds': preds}
            # return render(request, 'fileresults.html', context)
                #prediction_list = preds.tolist()
                prediction_list = ['Default' if pred == 1 else 'Non-default' for pred in preds]
                i = d['ID'] 
            # Zip the input data and the predictions
                results = list(zip(i.tolist(), prediction_list))
            # Pass the results and file name to the template
                return render(request, 'fileresults.html', {'results': results, 'file_name': file_name})
            
            elif 'button2' in request.POST:
                #file_name = form.cleaned_data['file_name']
                file = form.cleaned_data['file']
                file_name = file.name
                csv_filename = "default.csv"
                csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static", csv_filename)
                df = pd.read_csv(csv_path)
                #csv_path = "../static/default.csv"
                #df = pd.read_csv(csv_path)
                df['EDUCATION'].replace([0, 6], 5, inplace=True)
                df['EDUCATION'].replace(5, 4, inplace=True)
                df['MARRIAGE'].replace(0, 3, inplace=True)
                le = LabelEncoder()
                df.insert(2, 'Gender_encoded', le.fit_transform(df['SEX']))
                df.insert(3, 'education_encoded', le.fit_transform(df['EDUCATION']))
                df.insert(4, 'marriage_encoded', le.fit_transform(df['MARRIAGE']))
                df = df.drop('SEX', axis=1)
                df = df.drop('EDUCATION', axis=1)
                df = df.drop('MARRIAGE', axis=1)
                df.drop('ID',axis=1, inplace=True)
                X = df.iloc[:,:-1].values
                y = df.iloc[:,-1].values
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                scaler = RobustScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                # do something with the file
                # return render(request, 'success.html')
                d = pd.read_csv(file)
                copied_d = d.copy()
                copied_d['EDUCATION'].replace([0, 6], 5, inplace=True)
                copied_d['EDUCATION'].replace(5, 4, inplace=True)
                copied_d['MARRIAGE'].replace(0, 3, inplace=True)

                le = LabelEncoder()
                    # Encode the 'Sex' column and insert it at the 3rd position
                copied_d.insert(2, 'Gender_encoded', le.fit_transform(d['SEX']))
                    # Encode the 'Sex' column and insert it at the 3rd position
                copied_d.insert(3, 'education_encoded', le.fit_transform(d['EDUCATION']))
                copied_d.insert(4, 'marriage_encoded', le.fit_transform(d['MARRIAGE']))
                copied_d = copied_d.drop('MARRIAGE', axis=1)
                copied_d = copied_d.drop('SEX', axis=1)
                copied_d = copied_d.drop('EDUCATION', axis=1)
                copied_d.drop('ID',axis=1, inplace=True)
                X_1 = copied_d.iloc[:,:-1].values
                y_1 = copied_d.iloc[:,-1].values
                scaled_new_data = scaler.transform(X_1)
                #svm
                c=joblib.load('mlmodel/svc_model.joblib')
                scores = c.decision_function(scaled_new_data)
                new_threshold = -0.15
                preds = (scores >= new_threshold).astype(int)
                #preds = c.predict(scaled_new_data)
                report = classification_report(y_1, preds)
                cm = confusion_matrix(y_1, preds)
                plt.figure(figsize=(4,4))
                sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
                plt.title("Confusion Matrix-svm")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.tight_layout()
                 # Save the plot to a buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                # Encode the buffer as a base64 string
                image_png = buffer.getvalue()
                buffer.close()
                graphic = base64.b64encode(image_png)
                graphic = graphic.decode('utf-8')
                #logistic
                log=joblib.load('mlmodel/log_model.joblib')
                ans = log.predict(scaled_new_data)

                report1 = classification_report(y_1, ans)
                cm1 = confusion_matrix(y_1, ans)
                plt.figure(figsize=(4,4))
                sns.heatmap(cm1, annot=True, cmap="Blues", fmt="d", cbar=False)
                plt.title("Confusion Matrix-logistic regression")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.tight_layout()

                 # Save the plot to a buffer
                buffer1 = io.BytesIO()
                plt.savefig(buffer1, format='png')
                buffer1.seek(0)

                # Encode the buffer as a base64 string
                image_png1 = buffer1.getvalue()
                buffer1.close()
                graphic1 = base64.b64encode(image_png1)
                graphic1 = graphic1.decode('utf-8')
                #decision tree
                dt=joblib.load('mlmodel/tree_model.joblib')
                #print(type(dt)) 
                #arr= X_1.reshape(1,-1)
                #scale= scaler.transform(arr)
                a = dt.predict(scaled_new_data)
                
                
                #pred = dt.predict(scale)

                report2 = classification_report(y_1, a)
                cm2 = confusion_matrix(y_1, a)
                plt.figure(figsize=(4,4))
                sns.heatmap(cm2, annot=True, cmap="Blues", fmt="d", cbar=False)
                plt.title("Confusion Matrix-Deciison tree")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.tight_layout()

                 # Save the plot to a buffer
                buffer2 = io.BytesIO()
                plt.savefig(buffer2, format='png')
                buffer2.seek(0)

                # Encode the buffer as a base64 string
                image_png2 = buffer2.getvalue()
                buffer2.close()
                graphic2 = base64.b64encode(image_png2)
                graphic2 = graphic2.decode('utf-8')
                #random forest
                rf=joblib.load('mlmodel/rf_model.joblib')
                pr = rf.predict(scaled_new_data)

                report3 = classification_report(y_1, pr)
                cm3 = confusion_matrix(y_1, pr)
                plt.figure(figsize=(4,4))
                sns.heatmap(cm3, annot=True, cmap="Blues", fmt="d", cbar=False)
                plt.title("Confusion Matrix-random forest")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.tight_layout()

                 # Save the plot to a buffer
                buffer3 = io.BytesIO()
                plt.savefig(buffer3, format='png')
                buffer3.seek(0)

                # Encode the buffer as a base64 string
                image_png3 = buffer3.getvalue()
                buffer3.close()
                graphic3 = base64.b64encode(image_png3)
                graphic3 = graphic3.decode('utf-8') 
            return render(request, 'analyze.html', {'report': report, 'graphic': graphic, 'report1':report1, 'graphic1':graphic1, 'report2':report2, 'graphic2':graphic2, 'report3':report3, 'graphic3':graphic3})
            
        else:
            return HttpResponse("Upload failed")
    return render(request, 'file.html', {'form': form})
  


def contact(request): 
    return render(request, 'contact.html')

def result(request):
    csv_filename = "default.csv"
    csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static", csv_filename)
    df = pd.read_csv(csv_path)
    #csv_path = "../static/default.csv"
    #df = pd.read_csv(csv_path)
    df['EDUCATION'].replace([0, 6], 5, inplace=True)
    df['EDUCATION'].replace(5, 4, inplace=True)
    df['MARRIAGE'].replace(0, 3, inplace=True)
    le = LabelEncoder()
    df.insert(2, 'Gender_encoded', le.fit_transform(df['SEX']))
    df.insert(3, 'education_encoded', le.fit_transform(df['EDUCATION']))
    df.insert(4, 'marriage_encoded', le.fit_transform(df['MARRIAGE']))
    df = df.drop('SEX', axis=1)
    df = df.drop('EDUCATION', axis=1)
    df = df.drop('MARRIAGE', axis=1)
    df.drop('ID',axis=1, inplace=True)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    cls=joblib.load('mlmodel/svc_model.joblib')
    input_dict = {}
    input_dict = {
        'limit_bal': request.GET['limit_bal'],
        'gender': request.GET['gender'],
        'education': request.GET['education'],
        'marriage': request.GET['marriage'],
        'age': request.GET['age'],
        'pay_0': request.GET['pay_0'],
        'pay_2': request.GET['pay_2'],
        'pay_3': request.GET['pay_3'],
        'pay_4': request.GET['pay_4'],
        'pay_5': request.GET['pay_5'],
        'pay_6': request.GET['pay_6'],
        'bill_amt1': request.GET['bill_amt1'],
        'bill_amt2': request.GET['bill_amt2'],
        'bill_amt3': request.GET['bill_amt3'],
        'bill_amt4': request.GET['bill_amt4'],
        'bill_amt5': request.GET['bill_amt5'],
        'bill_amt6': request.GET['bill_amt6'],
        'pay_amt1': request.GET['pay_amt1'],
        'pay_amt2': request.GET['pay_amt2'],
        'pay_amt3': request.GET['pay_amt3'],
        'pay_amt4': request.GET['pay_amt4'],
        'pay_amt5': request.GET['pay_amt5'],
        'pay_amt6': request.GET['pay_amt6']
        }
        # reorder the input data based on the order of the training data
    input_list = [
            input_dict['limit_bal'],
            input_dict['gender'],
            input_dict['education'],
            input_dict['marriage'],
            input_dict['age'],
            input_dict['pay_0'],
            input_dict['pay_2'],
            input_dict['pay_3'],
            input_dict['pay_4'],
            input_dict['pay_5'],
            input_dict['pay_6'],
            input_dict['bill_amt1'],
            input_dict['bill_amt2'],
            input_dict['bill_amt3'],
            input_dict['bill_amt4'],
            input_dict['bill_amt5'],
            input_dict['bill_amt6'],
            input_dict['pay_amt1'],
            input_dict['pay_amt2'],
            input_dict['pay_amt3'],
            input_dict['pay_amt4'],
            input_dict['pay_amt5'],
            input_dict['pay_amt6']
        ]
    data_array = np.asarray(input_list)
    arr= data_array.reshape(1,-1)
    scaled_data = scaler.transform(arr)
    ans = cls.predict(scaled_data)
    #scores = cls.decision_function(scaled_data)
    #new_threshold = -0.45
            # probs = c.predict_proba(scaled_new_data)[:, 1]
        
        # apply the new threshold value
   # ans = (scores >= new_threshold).astype(int)
    #print(ans)
    finalans=''
    if(ans==1):
        finalans='The customer is probably going to DEFAULT'
    elif(ans==0):
        finalans = 'The custoner is a Non-Defaulter'
    return render(request, "result.html",{'result2':finalans})


