from django.shortcuts import render
import joblib

def index(request):
    return render(request,'index.html')


def predict(request):
    cls=joblib.load('finalmodel.sav')
    lis=[]
    lis.append(request.POST['Gender'])
    lis.append(request.POST['Married'])
    lis.append(request.POST['Dependents'])
    lis.append(request.POST['Education'])
    lis.append(request.POST['Self_Employed'])
    lis.append(request.POST['ApplicantIncome'])
    lis.append(request.POST['CoapplicantIncome'])
    lis.append(request.POST['LoanAmount'])
    lis.append(request.POST['Loan_Amount_Term'])
    lis.append(request.POST['Credit_History'])
    lis.append(request.POST['Property_Area'])
    ans=cls.predict([lis])


    g=request.POST['Gender']
    m=request.POST['Married']
    d=request.POST['Dependents']
    e=request.POST['Education']
    s=request.POST['Self_Employed']
    a=request.POST['ApplicantIncome']
    ca=request.POST['CoapplicantIncome']
    loan=request.POST['LoanAmount']



    

    if (g==0 and m==0 and d==0 and s==0):
        ans=0
    return render(request,"predict.html",{'ans':ans})
