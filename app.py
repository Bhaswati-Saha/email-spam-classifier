from flask import Flask,render_template,request
import numpy as np
import pickle
app=Flask(__name__)

cv = pickle.load(open('models/cv.pkl','rb'))
clf = pickle.load(open('models/clf.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['post'])
def predict():
    email=request.form.get('email')
    X_input=cv.transform([email]).toarray()#converting to numpy array
    print(X_input.shape)
    y_pred=clf.predict(X_input)
    if y_pred[0]==1:
        response=1#spam
    else:
        response=-1#not spam
    return render_template("index.html",response=response)
if __name__=="__main__":
      app.run(debug=True)
