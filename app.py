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
# print("Storing embeddings in Pinecone...")
# store_embeddings_in_pinecone()
# print("Embeddings stored successfully!")




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

        # Generate the RFP
        generated_rfp = generate_rfp(user_query)

        # Save the generated RFP to a file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"rfp_{timestamp}.txt")
        with open(output_file, 'w') as f:
            f.write(generated_rfp)

        # Return the generated RFP and a download link
        return jsonify({
            "rfp": generated_rfp,
            "download_link": f"/download_rfp/{timestamp}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_rfp/<timestamp>')
def download_rfp(timestamp):
    output_file = os.path.join(OUTPUT_DIR, f"rfp_{timestamp}.txt")
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)