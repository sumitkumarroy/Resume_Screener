from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_FOLDER = 'resumes'
ALLOWED_EXTENSIONS = {'pdf','dox'}

nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

def rank_resumes(resume_texts, job_description):
    job_desc_processed = preprocess(job_description)
    processed_resumes = [preprocess(text) for text in resume_texts.values()]
    documents = [job_desc_processed] + processed_resumes

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
    ranked = sorted(zip(resume_texts.keys(), scores), key=lambda x: x[1], reverse=True)
    return ranked

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    job_desc = request.form.get("job_desc", "")
    files = request.files.getlist("resumes")
    resume_texts = {}

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            text = extract_text(path)
            resume_texts[filename] = text
            os.remove(path)

    results = rank_resumes(resume_texts, job_desc)
    response = [{"name": name, "score": f"{round(score * 100, 2)}%"} for name, score in results]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
