import numpy as np
import base64
import cv2
from flask import Flask, render_template, request
from model.preprocessing import DataPipeline
from model.ResNet34 import ResNet34
from model.WordFinder import WordFinder

init_Base64 = 21
IMG_HEIGHT = 28
IMG_WIDTH = 28

categories = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',
              6:'G',7:'H',8:'I',9:'J',10:'K',
              11:'L',12:'M',13:'N',14:'O',15:'P',
              16:'Q',17:'R',18:'S',19:'T',20:'U',
              21:'V',22:'W',23:'X',24:'Y',25:'Z' }

nn = ResNet34(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))

modelo = nn.model_blueprint()


app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])


def index():
    word = []
    if request.method == 'POST':
        draw = request.form['url']
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)

        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        pipeline = DataPipeline(image, IMG_HEIGHT, IMG_WIDTH, 10)
        letters = pipeline.get_letters()

        letters = sorted(letters, key=lambda x: x[1])
        if len(letters)!=0: 
            predictions = modelo.predict(np.asarray([i[0] for i in letters]))
            for p in predictions:
                word.append(categories[np.argmax(p)])
        
    #palabras = WordFinder(word)
    print(word)
    palabras = ''.join(word)
    #print(palabras)


    return render_template('index.html', len=len(palabras), palabras=palabras)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', use_reloader = True, port=8080, debug = True)
