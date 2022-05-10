from flask import Flask, render_template, request
#import tensorflow.keras as keras
from keras.models import load_model
from keras.preprocessing import image
import keras
import PIL
import cv2
import numpy
from explainableai import ExplainableAI
import base64
import io


app = Flask(__name__)

dic = {0 : 'Covid Positive', 1 : 'Covid Negative'}

model = load_model('model.h5')

model.make_predict_function()
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
def predict_label(img_path):
	i = keras.utils.load_img(img_path, target_size=(64,64))
	j=i
	i = keras.utils.img_to_array(i)
	i = numpy.expand_dims(i,axis=0)
	i = i/255
	p = model.predict(i)
	xai = ExplainableAI(j,model)
	im = xai
	data = io.BytesIO()
	im.save(data, "JPEG")
	encoded_img_data = base64.b64encode(data.getvalue())
	#return dic[numpy.argmax(p[0])]
	return dic[numpy.argmax(p[0])],encoded_img_data



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/runtests", methods=['GET', 'POST'])
def runtest():
	return render_template("runtests.html")

@app.route("/output", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p,exai = predict_label(img_path)
		#p = predict_label(img_path)

	#return render_template("index.html", prediction = p, img_path = img_path)
	#return render_template("index.html", prediction = p, img_path = img_path, img_path_xai = exai)
	return render_template("covid19.html", prediction = p, img_path = img_path, img_data=exai.decode('utf-8'))
	#return render_template("index.html", prediction = p, img_path = img_path, img_data=exai.decode('utf-8'))
	


if __name__ =='__main__':
	#app.debug = True
	#app.run(debug=False,host='0.0.0.0')
	app.run()