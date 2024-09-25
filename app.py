from flask import Flask, request, jsonify
import subprocess
import json
import os

app = Flask(__name__)

def run_script(script_name, params):
    """ Run a Python script with parameters and capture its output """
    try:
        # Save parameters to a file
        with open('parameters.json', 'w') as f:
            json.dump(params, f)
        
        # Run the Python script
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            check=True
        )
        return {'success': True, 'output': result.stdout}
    except subprocess.CalledProcessError as e:
        return {'success': False, 'error': e.stderr}

@app.route('/simulate', methods=['POST'])
def simulate():
    params = request.json
    result = run_script('runSimulation.py', params)
    if result['success']:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': result['error']})

@app.route('/retrieve', methods=['POST'])
def retrieve():
    params = request.json
    result = run_script('runRetrieval.py', params)
    if result['success']:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': result['error']})

@app.route('/fuse', methods=['POST'])
def fuse():
    params = request.json
    result = run_script('runFusion.py', params)
    if result['success']:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': result['error']})

if __name__ == '__main__':
    # Ensure the necessary directories exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    app.run(debug=True)
