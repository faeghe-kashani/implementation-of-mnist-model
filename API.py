from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)


model = joblib.load('random_forest_mnist.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST' and 'image' in request.files:
        
        image = Image.open(request.files['image']).convert('L')  
        image_arr = np.asarray(image)
        image_arr = image_arr.reshape(1, -1)  


        
        prediction = model.predict(image_arr)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)