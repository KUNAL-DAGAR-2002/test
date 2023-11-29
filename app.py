from flask import Flask, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model("apple_vgg16.h5")

app = Flask(__name__)
@app.route('/',methods=['POST'])
def served_model(): 
    request_data = request.get_json(force=True)
    img = request_data["img"]
    img = np.array(img).reshape(-1,224,224,3)
    return ("{}".format(["scab","rot","Healthy"][model.predict(img).argmax()]))

if __name__ == "__main__":
    app.run()