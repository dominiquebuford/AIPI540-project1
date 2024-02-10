from flask import Blueprint, request, jsonify
from .classify_images import preprocess_image, classify_image
from .ai_response import generate_ai_response

main = Blueprint('main', __name__)

@main.route('/process_request', methods=['POST'])
def process_request():
    child_name = request.form.get('childName', 'Anonymous')
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    results = []

    for uploaded_file in files:
        try:
            img_batch = preprocess_image(uploaded_file)
            class_label = classify_image(img_batch)
            results.append(class_label)
        except Exception as e:
            return jsonify({'error': f'Failed to process {uploaded_file.filename}. Error: {str(e)}'}), 500

    # Generate a story based on the child's name and the image classification results
    story = generate_ai_response(child_name, results)

    return jsonify({'story': story})

# TODO: Make sure to register the Blueprint in your Flask app setup
