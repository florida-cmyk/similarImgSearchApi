import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
import keras
import keras.utils as image
import requests
import json, codecs
import io

from flask import Flask, request, Response
from scipy.spatial import distance
from PIL import Image
from sklearn.decomposition import PCA
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model


app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello world'

@app.route('/api/<path:url>', methods=['GET'])
def api(url):
    response = requests.get(url)
    with open('dataset/0.jpg', 'wb') as file:
        file.write(response.content)
    model = keras.applications.VGG16(weights='imagenet', include_top=True)

    def load_image(path):
        img = image.load_img(path, target_size=model.input_shape[1:3])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x

    img, x = load_image("dataset/0.jpg")
    predictions = model.predict(x)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    feat = feat_extractor.predict(x)

    images_path = 'dataset'
    image_extensions = ['.jpg', '.png', '.jpeg'] 
    max_num_images = 10000

    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
    if max_num_images < len(images):
        images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]

    tic = time.perf_counter ()

    features = []
    print(features)
    input()
    for i, image_path in enumerate(images):
        if i % 500 == 0:
            toc = time.perf_counter ()
            elap = toc-tic;
            print("анализ изображения %d / %d. Время: %4.4f секунды." % (i, len(images),elap))
            tic = time.perf_counter ()
        img, x = load_image(image_path);
        feat = feat_extractor.predict(x)[0]
        features.append(feat)

    print("Завершено извлечение особенностей для %d изображений" % len(images))

    features = np.array(features)

    pca = PCA(n_components=100)
    pca.fit(features)
    pca_features = pca.transform(features)

    def get_closest_images(query_image_idx, num_results=5):
        distances = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
        return idx_closest

    def get_concatenated_images(indexes, thumb_height):
        thumbs = []
        for idx in indexes:
            img = image.load_img(images[idx])
            img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
            thumbs.append(img)
        concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
        return concat_image

    query_image_idx = 0
    idx_closest = get_closest_images(query_image_idx, 1)
    query_image = get_concatenated_images([query_image_idx], 300)
    results_image = get_concatenated_images(idx_closest, 200)    
    img = Image.fromarray(results_image)
    
    # сохранение изображения в формате JPEG в байтовом представлении
    img_bytearr = io.BytesIO()
    img.save(img_bytearr, format='JPEG')
    img_bytearr = img_bytearr.getvalue()
    
    # отображение изображения в формате JPEG
    return Response(img_bytearr, mimetype='image/jpeg')
   
   
if __name__ == '__main__':
    app.run()
    # app.run(host='10.0.27.52', port='3128')



