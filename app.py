from flask import Flask, jsonify, request
import requests
from app.train import train
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		company = request.get_json()
		# print(company)
		pred = train(company)
		return jsonify({'prediction':str(pred)})
	return 'OK'

if __name__ == "__main__":
	app.run()