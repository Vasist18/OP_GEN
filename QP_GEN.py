import streamlit as st
from openai import OpenAI
import pytesseract
import cv2
import numpy as np
import fitz  # PyMuPDF
import json
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ==========================================
# 1. PAGE CONFIG & UI SETUP
# ==========================================
st.set_page_config(page_title="AI Exam Generator", layout="wide", page_icon="📝")

# Custom CSS for a modern dashboard look
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); }
    .glass-card { background: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# Replace with your actual key or use st.secrets
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Initialize Session State
if 'paper_data' not in st.session_state:
    st.session_state.paper_data = None

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
def extract_text_from_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(gray).strip()

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    return text.strip()

def generate_questions(context_text, config_data):
    system_prompt = """
    You are an expert academic content creator. Generate an exam paper based ONLY on the provided text.
    You must output a strictly valid JSON object with this exact structure:
    {
        "questions": {
            "vsa": [{"q": "Question text", "marks": 1}],
            "sa": [{"q": "Question text", "marks": 3}],
            "la": [{"q": "Question text", "marks": 5}],
            "mcq": [{"q": "Question text", "options": ["A", "B", "C", "D"], "marks": 1}]
        }
    }
    """
    prompt = f"Exam Config: {json.dumps(config_data)}\n\nSource Material:\n{context_text}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return json.loads(response.choices[0].message.content)

def chat_modify_questions(current_data, command):
    system_prompt = "You are an AI teaching assistant. Modify the provided JSON question paper based on the user's instructions. Return ONLY the updated strict JSON object."
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current Paper: {json.dumps(current_data)}\n\nInstruction: {command}"}
        ]
    )
    return json.loads(response.choices[0].message.content)

def draw_page_border(canvas, doc):
    canvas.saveState()
    canvas.rect(0.5 * inch, 0.5 * inch, A4[0] - 1 * inch, A4[1] - 1 * inch)
    canvas.restoreState()

def create_pdf_buffer(paper_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    
    header_style = ParagraphStyle('Header', parent=styles['Normal'], fontSize=16, alignment=1, spaceAfter=10, fontName='Helvetica-Bold')
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontSize=12, alignment=1, spaceAfter=20)
    section_style = ParagraphStyle('Section', parent=styles['Normal'], fontSize=14, fontName='Helvetica-Bold', spaceBefore=15, spaceAfter=10)
    q_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=11, spaceBefore=5, spaceAfter=5)

    story = []
    story.append(Paragraph(paper_data.get('school', 'School Name'), header_style))
    meta_text = f"Subject: {paper_data.get('subject', '')} | Class: {paper_data.get('grade', '')} | Max Marks: {paper_data.get('marks', '')}"
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 0.2 * inch))

    sections_map = {
        "vsa": "SECTION A - Very Short Answer",
        "sa": "SECTION B - Short Answer",
        "la": "SECTION C - Long Answer",
        "mcq": "SECTION D - Multiple Choice"
    }

    q_num = 1
    q_dict = paper_data.get("questions", {})
    
    for key, title in sections_map.items():
        if key in q_dict and q_dict[key]:
            story.append(Paragraph(title, section_style))
            for item in q_dict[key]:
                if key == "mcq":
                    opts = "  ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(item.get('options', []))])
                    text = f"{q_num}. {item['q']} [{item['marks']}M]<br/>{opts}"
                else:
                    text = f"{q_num}. {item['q']} [{item['marks']}M]"
                story.append(Paragraph(text, q_style))
                q_num += 1

    doc.build(story, onFirstPage=draw_page_border, onLaterPages=draw_page_border)
    buffer.seek(0)
    return buffer

# ==========================================
# 3. DASHBOARD UI
# ==========================================
st.title("📝 AI Question Paper Generator")

col1, col2 = st.columns([1, 2])

# --- LEFT COLUMN: SETUP ---
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("1. Exam Setup")
    
    school = st.text_input("School Name")
    subject = st.text_input("Subject")
    grade = st.text_input("Class / Grade")
    marks = st.number_input("Maximum Marks", min_value=10, max_value=100, value=50)
    
    uploaded_file = st.file_uploader("Upload Study Material", type=["pdf", "png", "jpg", "jpeg"])
    
    if st.button("Generate Paper", type="primary", use_container_width=True):
        if not uploaded_file or not school or not subject:
            st.error("Please fill all details and upload a file.")
        else:
            with st.spinner("Extracting text and generating questions..."):
                file_bytes = uploaded_file.read()
                if uploaded_file.name.endswith('.pdf'):
                    text = extract_text_from_pdf(file_bytes)
                else:
                    text = extract_text_from_image(file_bytes)
                
                config = {"school": school, "subject": subject, "grade": grade, "marks": marks}
                ai_result = generate_questions(text, config)
                
                # Merge metadata into the final JSON
                ai_result.update(config)
                st.session_state.paper_data = ai_result
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT COLUMN: PREVIEW & CHAT ---
with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("2. Live Preview & Edit")
    
    if st.session_state.paper_data:
        data = st.session_state.paper_data
        
        # Display the paper layout
        st.markdown(f"<h3 style='text-align: center;'>{data.get('school')}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Subject:** {data.get('subject')} | **Class:** {data.get('grade')} | **Marks:** {data.get('marks')}")
        st.divider()
        
        q_dict = data.get("questions", {})
        q_num = 1
        
        if q_dict.get('mcq'):
            st.markdown("#### SECTION - Multiple Choice")
            for item in q_dict['mcq']:
                st.write(f"**{q_num}. {item['q']}** [{item['marks']}M]")
                st.caption(" | ".join(item['options']))
                q_num += 1
                
        if q_dict.get('sa'):
            st.markdown("#### SECTION - Short Answer")
            for item in q_dict['sa']:
                st.write(f"**{q_num}. {item['q']}** [{item['marks']}M]")
                q_num += 1

        st.divider()
        
        # Chat Modifier
        st.subheader("Modify with AI")
        chat_cmd = st.text_input("Tell AI to change the paper (e.g., 'Add 2 more MCQs about gravity')")
        if st.button("Update Paper"):
            with st.spinner("Modifying..."):
                updated_questions = chat_modify_questions(st.session_state.paper_data, chat_cmd)
                st.session_state.paper_data['questions'] = updated_questions.get('questions', {})
                st.rerun()
                
        # Export PDF
        st.subheader("3. Export")
        pdf_buffer = create_pdf_buffer(st.session_state.paper_data)
        st.download_button(
            label="📄 Download Ready-to-Print PDF",
            data=pdf_buffer,
            file_name=f"{data.get('subject')}_Paper.pdf",
            mime="application/pdf",
            type="primary"
        )
    else:
        st.info("Fill out the setup form on the left to generate a live preview.")
        
    st.markdown('</div>', unsafe_allow_html=True)