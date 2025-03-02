from flask import Flask, render_template, request, jsonify
import os
import datetime
from rfp_generator import generate_rfp_from_query

app = Flask(__name__)

# Set the path to save the RFP outputs
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_rfp', methods=['POST'])
def generate_rfp():
    user_query = request.form['query']
    
    # Generate RFP text based on the user's query
    generated_rfp = generate_rfp_from_query(user_query)
    
    # # Save the generated RFP to a text file
    # filename = f"{OUTPUT_DIR}/rfp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    # with open(filename, 'w') as f:
    #     f.write(generated_rfp)
    
    return jsonify({'rfp': generated_rfp})

if __name__ == '__main__':
    app.run(debug=True)
