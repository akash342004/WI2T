Here is a complete README.md for your project:


---

# AI-Powered Image to Text Extraction & Language Correction Web App

This web application extracts text from uploaded images, detects the language (English, Hindi, Telugu), and offers formatting using transformer-based models like GPT-2 and IndicBART. Users can download the result as text or PDF.

---

## Features

- Upload image files (JPG, PNG, JPEG, BMP).
- Detect language using a custom deep learning model.
- Extract text using OCR (EasyOCR).
- Correct and format text using:
  - **GPT-2** for English
  - **IndicBART** for Hindi and Telugu
- Download output as:
  - **.txt** file
  - **.pdf** file
- Stylish UI with animated feedback

---

## Demo

> You can try the app live here (after deployment):  
> **[Your Render Link Goes Here]**

---

## Folder Structure

project/ │ ├── app.py                  # Main Flask backend ├── detector2.keras         # Trained Keras language detection model ├── requirements.txt ├── Procfile ├── runtime.txt ├── uploads/                # Temporary uploaded images (excluded in .gitignore) │ ├── templates/              # HTML templates (Jinja2) │   ├── intro.html │   ├── detect.html │   ├── extracted.html │   ├── end.html │   ├── error.html │   ├── close.html │   └── hindi_telugu.html │ ├── static/ │   └── images/ │       ├── background2.jpg │       └── thank.jpg │ └── .gitignore

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the App

python app.py

Visit http://127.0.0.1:5000 in your browser.


---

Deploying on Render (Free Hosting)

1. Push this repo to GitHub.


2. Go to https://render.com.


3. Click “New Web Service”.


4. Connect your GitHub repo.


5. Set:

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app



6. Wait for it to deploy.




---

Dependencies

Flask

TensorFlow

TensorFlow Hub

Transformers (Hugging Face)

EasyOCR

PyTorch

ReportLab

Pillow

NumPy, Matplotlib

SpellChecker


See requirements.txt for versions.


---

Screenshots

> Add screenshots of UI pages here (intro, detect, result, download).




---

License

MIT License


---

Author

Akash
Undergraduate Major Project | Veltech University
MS in Data Analytics & Information Systems – Texas State University


---

Acknowledgments

Hugging Face Transformers

ai4bharat/IndicBART

EasyOCR

TensorFlow + Keras


---

Let me know if you want this bundled in a ZIP or uploaded to a GitHub repo automatically.
