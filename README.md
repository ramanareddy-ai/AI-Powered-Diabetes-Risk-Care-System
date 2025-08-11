# 🩺 AI-Powered Diabetes Risk & Care System

An interactive **Streamlit** application that predicts a person's diabetes risk using **Machine Learning** (Logistic Regression) and **OpenAI GPT** for educational explanations.  
Generates a **beautiful, patient-friendly PDF report** with:  
- Patient-entered metrics  
- Risk score & category  
- AI-generated plain-language explanation  
- Likely contributing factors  
- Suggested next steps for care  
- Logo, patient details, and a visual risk chart  

---

## Features
- **Interactive Web UI** powered by [Streamlit](https://streamlit.io/).
- **Machine Learning model** trained on Pima Indians Diabetes Dataset.
- **AI Explanation** via OpenAI API (compassionate, non-medical educational tone).
- **Beautiful PDF Report**:
  - Includes patient info (Name, Case ID).
  - Unicode font support (handles emojis & non-Latin scripts).
  - Visual risk bar chart.
  - Auto-saves reports to `/reports` folder with timestamps.
- **Flexible CSV ingestion** — falls back to synthetic dataset if missing.

---

##  Project Structure
```
AI_Powered_Diabetes_Risk_Care_System_ML/
├── app/
│   ├── assets/
│   │   └── fonts/
│   │       └── DejaVuSans.ttf
│   ├── core/
│   │   ├── data.py
│   │   └── llm.py
│   ├── tabs/
│   │   ├── diagnosis.py
│   │   └── results.py
│   └── main.py
├── requirements.txt
└── README.md
```

---

##  Installation & Setup

### 1 Clone or Download
```bash
git clone https://github.com/yourusername/AI_Powered_Diabetes_Risk_Care_System_ML.git
cd AI_Powered_Diabetes_Risk_Care_System_ML
```

### 2 Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### 3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4 Add OpenAI API Key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5 Add Unicode Font (already included)
Ensure `app/assets/fonts/DejaVuSans.ttf` is present.  
If missing:
```bash
mkdir -p app/assets/fonts
curl -L -o app/assets/fonts/DejaVuSans.ttf   https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSans.ttf
```

---

## ▶️ Run the App
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example PDF Output
The generated PDF includes:
1. **Patient Information**  
2. **Metrics Table**  
3. **Risk Category & Score**  
4. **Visual Risk Chart**  
5. **AI Explanation** (educational only)  
6. **Likely Contributing Factors**  
7. **Next Steps for Care**  

*(Sample Screenshot)*  
![Sample PDF Screenshot](app/assets/sample_pdf.png)

---

## Tech Stack
- **Python 3.10+**
- **Streamlit** (UI)
- **scikit-learn** (ML Model)
- **pandas / numpy** (Data Handling)
- **matplotlib** (Charts)
- **fpdf2** (PDF Reports)
- **OpenAI API** (Educational Explanations)

---

## Disclaimer
This app is **for educational purposes only**.  
It is **not** a medical diagnosis tool. Always consult a qualified healthcare professional for medical concerns.

---

## License
MIT License © 2025 Ramana Reddy
