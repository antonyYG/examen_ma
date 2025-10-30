from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Cargar modelo
model = load_model("modelo_mnist.h5")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Obtener archivo
        file = request.files['file']
        image = Image.open(file).convert('L')  
        image = image.resize((28, 28))
        image = np.array(image).astype('float32') / 255.0
        image = image.reshape(1, 28, 28, 1)
        
        # Predecir
        prediction = np.argmax(model.predict(image))
        return jsonify({'prediccion': int(prediction)})
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
