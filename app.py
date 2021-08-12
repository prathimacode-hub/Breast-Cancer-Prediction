import os
import joblib
classifier = joblib.load(r'breastcancer.pkl')

# importing Flask and other modules
from flask import Flask, request, render_template 
import numpy as np
  
# Flask constructor
app = Flask(__name__)   
  
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods = ["GET", "POST"])
def breast_cancer():
    if request.method == "POST":
       radius_mean = request.form.get("radius_mean")
       perimeter_se = request.form.get("perimeter_se")
       p = classifier.predict(np.array([radius_mean, perimeter_se]).reshape(-2,2))
       if p == 0:
           output = "You're suffering from benign breast cancer. Take precautions."
           return render_template('index.html', output=output)
       elif p == 1:
           output = "You're suffering from malignant cancer. Be careful and focus on your health."
           return render_template('index.html', output=output)
        
    return render_template('index.html')
  
if __name__=='__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port)
