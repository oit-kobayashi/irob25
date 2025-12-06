from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/hello", methods=['POST', 'GET'])
def hello():
    return 'world'


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
