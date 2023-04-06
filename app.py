from flask import Flask, render_template, request, send_file
import cv2
import tensorflow as tf
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths

import clip
import numpy as np


app= Flask(__name__)

bp_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

@app.route('/', methods=['POST'])
def remove():
    content= request.json
    image_path=content['url']
    image=cv2.imread(image_path)
    prediction = bp_model.predict_single(image) 
    mask = prediction.get_mask(threshold=0.05).numpy().astype(np.uint8)
    new_mask = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite('output.jpg', new_mask)
    if request.method=='POST':
        return send_file('output.jpg', mimetype='image/jpeg')
    else:
        return send_file('output.jpg', mimetype='image/jpeg')


if __name__=='__main__':
    app.run(debug=True)
