# -*- coding: utf-8 -*- 
# @Time : 1/11/24 16:01 
# @Author : ANG

from flask import Flask, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
from Models.generate import Generate

app = Flask(__name__)

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'OutPut_Image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传和输出目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('WebUI', 'UI.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'contentImage' not in request.files or 'styleImage' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    content_file = request.files['contentImage']
    style_file = request.files['styleImage']
    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if content_file and style_file and allowed_file(content_file.filename) and allowed_file(style_file.filename):
        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)
        content_file.save(os.path.join(UPLOAD_FOLDER, content_filename))
        style_file.save(os.path.join(UPLOAD_FOLDER, style_filename))

        # 进行风格迁移
        try:
            generator = Generate("light", os.path.join(UPLOAD_FOLDER, style_filename),
                                 os.path.join(UPLOAD_FOLDER, content_filename),
                                 os.path.join(OUTPUT_FOLDER, 'output.jpg'), (480, 600))
            output = generator.train(num_epochs=200, lr_decay_epoch=250)
            output_image = generator.postprocess(output)
            output_image.save(os.path.join(OUTPUT_FOLDER, 'output.jpg'))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        return jsonify({'message': 'File processed', 'output_file': 'output.jpg'})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/check_image')
def check_image():
    if os.path.exists(os.path.join(OUTPUT_FOLDER, 'output.jpg')):
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})


@app.route('/OutPut_Image/<filename>')
def uploaded_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
