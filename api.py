from flask import Flask, jsonify, make_response, request 

app = Flask(__name__)

@app.route('/score', methods=['POST'])


