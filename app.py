import argparse

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pandas as pd
import prophet_model

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        processed_data = prophet_model.main_process(json_data, socketio)
        processed_data['month'] = processed_data['month'].dt.strftime('%Y-%m-%d')

        return jsonify(processed_data.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    socketio.run(app, host="80.209.240.170", port=5000)
