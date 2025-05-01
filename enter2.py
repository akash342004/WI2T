from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from numpy import *
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import numpy as np
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import easyocr
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from spellchecker import SpellChecker
import warnings
from PIL import Image
import cv2

warnings.filterwarnings("ignore")

app = Flask('__name__')

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
path_to_upload = None
global_text = ""
count = 0
lang = None

# Initialize language models
language_models = {
    'en': {'model': None, 'tokenizer': None},
    'hi': {'model': None, 'tokenizer': None},
    'te': {'model': None, 'tokenizer': None}
}

def load_language_models():
    """Initialize all language models"""
    try:
        # English model (GPT-2)
        if not language_models['en']['model']:
            language_models['en']['tokenizer'] = AutoTokenizer.from_pretrained("gpt2")
            language_models['en']['model'] = AutoModelForSeq2SeqLM.from_pretrained("gpt2")
        
        # Hindi and Telugu model (IndicBART)
        if not language_models['hi']['model']:
            indicbart_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False)
            indicbart_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")
            
            language_models['hi']['tokenizer'] = indicbart_tokenizer
            language_models['hi']['model'] = indicbart_model
            language_models['te']['tokenizer'] = indicbart_tokenizer
            language_models['te']['model'] = indicbart_model
            
    except Exception as e:
        app.logger.error(f"Error loading language models: {str(e)}")

# Load models during startup
load_language_models()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_text(text, line_length=80):
    lines = []
    current_line = ""
    for word in text.split():
        if len(current_line) + len(word) + 1 <= line_length:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

# Load detection model
custom_objects = {'KerasLayer': hub.KerasLayer}
model = load_model(r'D:\AKASH\FINAL YEAR PROJECT WEB BASED IMAGE TO TEXT CONVERSION USING ADVANCED DEEP LEARNING\detector2.keras', custom_objects=custom_objects)

@app.route('/', methods=['GET', 'POST'])
def intro_start():
    return render_template("intro.html")

@app.route('/langauge_detect', methods=['GET', 'POST'])
def langauge():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            
            f = request.files['file']
            if f.filename == '':
                return redirect(request.url)
                
            if not allowed_file(f.filename):
                return render_template("error.html", n="Invalid file type. Please upload PNG, JPG, or JPEG")

            # Secure save with unique filename
            filename = secure_filename(f.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(file_path)

            global path_to_upload
            path_to_upload = file_path

            # Process image with PIL
            img = Image.open(file_path)
            
            # Convert image modes
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            
            # Resize and normalize
            resize = tf.image.resize(img_array, (256, 256))
            resize = resize / 255.0
            img_input = tf.expand_dims(resize, 0)
            
            # Predict
            predictions = model.predict(img_input)
            predicted_labels = np.argmax(predictions, axis=1)

            ref = {0: "English", 1: "Hindi", 2: "Telugu"}
            res = ref[predicted_labels[0]]

            global lang
            lang = {'English': 'en', 'Hindi': 'hi', 'Telugu': 'te'}.get(res)
                
            return render_template("detect.html", n=res)
            
    except Exception as e:
        app.logger.error(f"Image processing error: {str(e)}")
        return render_template("error.html", n=f"Error processing image: {str(e)}")

def extract_ocr():
    global lang
    if lang is None:
        return "Language is not Identified"

    try:
        logging.getLogger().setLevel(logging.ERROR)
        reader = easyocr.Reader([lang])
        results = reader.readtext(path_to_upload)

        global count, global_text
        count = 0
        final_text = ""
        for (bbox, text, prob) in results:
            final_text += " " + text
            count += len(text)

        global_text = final_text
        return final_text
    except Exception as e:
        app.logger.error(f"OCR Error: {str(e)}")
        return f"OCR Processing Error: {str(e)}"

@app.route('/extraction', methods=['GET', 'POST'])
def extract_text_language():
    res = extract_ocr()
    if "Error" in res:
        return render_template("error.html", n=res)
    return render_template("extracted.html", n=res, lang=lang.upper())

def correct_text(text, lang_code):
    """Generic text correction using appropriate model"""
    try:
        if lang_code not in language_models or not language_models[lang_code]['model']:
            return text
            
        tokenizer = language_models[lang_code]['tokenizer']
        model = language_models[lang_code]['model']
        
        if lang_code == 'en':
            # English correction with GPT-2
            input_ids = tokenizer.encode(text, return_tensors="pt")
            corrected_ids = model.generate(
                input_ids,
                max_length=count,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                attention_mask=torch.ones(input_ids.shape),
                pad_token_id=tokenizer.eos_token_id
            )
            return tokenizer.decode(corrected_ids[0], skip_special_tokens=True)
        else:
            # Hindi/Telugu correction with IndicBART
            inputs = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            generated_tokens = model.generate(
                **inputs,
                max_length=count,
                num_beams=5,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
            return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
    except Exception as e:
        app.logger.error(f"Text Correction Error ({lang_code}): {str(e)}")
        return text

@app.route('/gpt', methods=['GET', 'POST'])
def extract_text_gpt():
    global lang, global_text
    if lang not in ['en', 'hi', 'te']:
        return render_template("error.html", n="Language not supported for advanced correction")
    
    global_text = correct_text(global_text, lang)
    global_text = format_text(global_text)
    return render_template("end.html")

@app.route('/normal', methods=['GET', 'POST'])
def extract_text_normal():
    global lang, global_text
    global_text = format_text(global_text)
    return render_template("end.html")

def save_text_to_pdf(text, output_file="output.pdf"):
    try:
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        paragraph = Paragraph(text, styles["Normal"])
        story.append(paragraph)
        doc.build(story)
    except Exception as e:
        app.logger.error(f"PDF Generation Error: {str(e)}")

def save_text_to_text_file(text, output_file="output.txt"):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)
    except Exception as e:
        app.logger.error(f"Text File Save Error: {str(e)}")

@app.route('/pdf', methods=['GET', 'POST'])
def pdf():
    save_text_to_pdf(global_text)
    return render_template("close.html")

@app.route('/text', methods=['GET', 'POST'])
def text():
    save_text_to_text_file(global_text)
    return render_template("close.html")

if __name__ == "__main__":
    app.run(debug=False)