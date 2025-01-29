from flask import Flask, request, jsonify, Response, send_file
import os
import logging
from main import *

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for cross-origin requests (optional)
from flask_cors import CORS
CORS(app)

# Route to handle task execution
@app.route('/run', methods=['POST'])
def run_task():
    task_description = request.args.get('task', '')  # Get task from query params
    if not task_description:
        return jsonify({"error": "Task description is missing."}), 400

    try:
        result = execute_task(task_description)
        return jsonify({"result": result}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
      
# Route to read files
@app.route('/read', methods=['GET'])
def read_file():
    file_path = request.args.get('path')  # Get file path from query parameters
    if not file_path:
        return jsonify({"error": "File path is missing."}), 400
    if not os.path.exists(file_path):
        return Response("", status=404)  # File not found
    with open(file_path, 'r') as file:
        return Response(file.read(), content_type='text/plain', status=200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
