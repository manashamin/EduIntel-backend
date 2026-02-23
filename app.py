from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import numpy as np
import re
import pytesseract
from pdf2image import convert_from_bytes

# ---------------- CONFIG ----------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = Flask(__name__)
CORS(app)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- TEXT EXTRACTION ----------------

def extract_text(pdf_file):
    text = ""

    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        if len(text.strip()) > 50:
            return text

    except Exception as e:
        print("Normal extraction failed:", e)

    # OCR fallback
    print("Switching to OCR...")
    pdf_file.seek(0)
    images = convert_from_bytes(pdf_file.read())

    for image in images:
        ocr_text = pytesseract.image_to_string(image)
        text += ocr_text + "\n"

    return text

# ---------------- SPLIT QUESTIONS ----------------

def split_answers(text):
    pattern = r'(?:\bQ?\d+\.)'
    parts = re.split(pattern, text)

    cleaned = []
    for part in parts:
        part = part.strip()
        if len(part) > 20:
            cleaned.append(part)

    return cleaned

# ---------------- ANALYZE ROUTE ----------------
CORS(app)

@app.route("/")
def home():
    return "Backend is running successfully"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "teacher" not in request.files:
            return jsonify({"error": "Teacher key not uploaded"}), 400

        teacher_pdf = request.files["teacher"]
        student_pdfs = request.files.getlist("students")

        if len(student_pdfs) == 0:
            return jsonify({"error": "No student files uploaded"}), 400

        teacher_text = extract_text(teacher_pdf)
        teacher_answers = split_answers(teacher_text)

        if len(teacher_answers) == 0:
            return jsonify({"error": "Teacher answers could not be extracted"}), 400

        all_results = []

        # Encode teacher answers once (optimization)
        teacher_embeddings = model.encode(teacher_answers, convert_to_tensor=True)

        for student_pdf in student_pdfs:

            student_text = extract_text(student_pdf)
            student_answers = split_answers(student_text)

            student_result = {
                "pdf_name": student_pdf.filename,
                "questions": [],
                "performance_score": 0  # renamed
            }

            similarities = []

            if len(student_answers) == 0:
                student_result["performance_score"] = 0
                all_results.append(student_result)
                continue

            # Encode student answers once
            student_embeddings = model.encode(student_answers, convert_to_tensor=True)

            for i in range(min(len(teacher_answers), len(student_answers))):

                similarity = util.cos_sim(
                    teacher_embeddings[i],
                    student_embeddings[i]
                ).item()

                similarity = float(similarity)

                # 🔥 Strong unrelated filter
                if similarity < 0.20:
                    similarity = 0

                similarities.append(similarity)

                percent_score = round(similarity * 100, 2)

                if percent_score >= 75:
                    status = "Strong Understanding"
                elif percent_score >= 45:
                    status = "Partial Understanding"
                elif percent_score > 0:
                    status = "Weak Understanding"
                else:
                    status = "Not Related"

                student_result["questions"].append({
                    "question_number": i + 1,
                    "similarity_percent": percent_score,
                    "status": status
                })

            if similarities:
                overall = (sum(similarities) / len(similarities)) * 100
            else:
                overall = 0

            student_result["performance_score"] = round(overall, 2)

            all_results.append(student_result)

        return jsonify({
            "success": True,
            "data": all_results
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)