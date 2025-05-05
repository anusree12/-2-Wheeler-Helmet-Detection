
from flask import Flask, render_template, request, redirect, url_for
import os
from detect_helmet import run_detection

app = Flask(__name__, template_folder="templates")

# Folder to store uploaded files
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'avif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Check for valid file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']
    #print(".......image_file..........", image_file)

    if image_file.filename == '':
        return redirect(request.url)

    if image_file and allowed_file(image_file.filename):
        #print(".......image_file111..........", image_file)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)
        #print(".......image_path..........", image_path)

        # Get log and image path
        result_text, result_image_relative_path = run_detection(image_path)
        #print(".......result_image_relative_path..........", result_image_relative_path)

        result_image_path = result_image_relative_path.replace("static/", "")  # just "uploads/output.jpg"
        #print(".......result_image_path..........", result_image_path)

        return render_template('index.html', result_image=result_image_path, result_text=result_text)

    return "File type not allowed. Please upload a valid image file."



if __name__ == '__main__':
    app.run(debug=True , port= 5050)