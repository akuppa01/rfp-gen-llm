from flask import Flask, render_template, request, jsonify, send_file
import os
import datetime
from rfp_generator import generate_rfp, store_embeddings_in_pinecone

app = Flask(__name__)

# Set the path to save the RFP outputs
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Ensure embeddings are stored on startup
print("Storing embeddings in Pinecone...")
# store_embeddings_in_pinecone()
print("Embeddings stored successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_rfp', methods=['POST'])
def generate_rfp_endpoint():
    try:
        # Get the user query from the JSON body
        data = request.get_json()
        user_query = data.get('query')

        if not user_query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Generate the RFP based on the query
        generated_rfp, file_path = generate_rfp(user_query)

        # Save the generated RFP to a text file
        filename = f"{OUTPUT_DIR}/rfp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(generated_rfp)

        return jsonify({
            "message": "RFP generated successfully!",
            "generated_rfp": generated_rfp,
            "file_path": file_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_rfp', methods=['GET'])
def download_rfp():
    try:
        file_path = "./outputs/generated_rfp.txt"
        
        if not os.path.exists(file_path):
            return jsonify({"error": "No RFP file found"}), 404
        
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
