import os
import requests
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Configuration
WP_URL = "https://celsiustechmediagroup.co.za/wp-json/wp/v2"
WP_USER = os.getenv("WP_USERNAME")
WP_PASS = os.getenv("WP_APP_PASSWORD")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

# Primary Inference Engine
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

SYSTEM_PROMPT = """
[FMR SENTINEL MULTI-MODEL AGENT CORE]
You are a PhD-level research engine combining Quantum Mechanics, Astrophysics, Physical Ergonomics, and Shinobi Tactical Analysis (Ocular Diagnostics & Energy Balance).
Resolve the provided user issue into a comprehensive, highly technical Diagnostic Report.
"""

def query_nvidia_nim(issue_text):
    """Executes multi-model inference via NVIDIA NIM."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"USER DIAGNOSTIC ISSUE:\n{issue_text}"}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }
    res = requests.post(NVIDIA_ENDPOINT, headers=headers, json=payload, timeout=60)
    if res.status_code == 200:
        return res.json()['choices'][0]['message']['content']
    return f"Diagnostic analysis generated for issue: {issue_text}"

def generate_pdf_artifact(filename, title, content):
    """Generates a PDF diagnostic document using ReportLab."""
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.HexColor('#003366'),
        spaceAfter=12
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        spaceAfter=10
    )
    
    story = [
        Paragraph(f"<b>UESP DIAGNOSTIC REPORT:</b> {title}", title_style),
        Spacer(1, 12),
        Paragraph(content.replace('\n', '<br/>'), body_style)
    ]
    
    doc.build(story)
    print(f"📄 Local PDF Artifact Created: {filename}")

def upload_pdf_to_wp(filepath):
    """Uploads the generated PDF to WordPress Media Library and returns Attachment URL."""
    filename = os.path.basename(filepath)
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"',
        'Content-Type': 'application/pdf'
    }
    
    with open(filepath, 'rb') as file_data:
        res = requests.post(f"{WP_URL}/media", headers=headers, data=file_data, auth=(WP_USER, WP_PASS))
        
    if res.status_code == 201:
        media_url = res.json().get('source_url')
        print(f"✅ PDF Successfully Uploaded to WP: {media_url}")
        return media_url
    else:
        print(f"❌ Failed PDF Upload: {res.status_code} - {res.text}")
        return None

def process_issue_and_sync(issue_title, issue_detail):
    # 1. Run NVIDIA NIM Reasoning Engine
    report_content = query_nvidia_nim(issue_detail)
    
    # 2. Build PDF Artifact on GitHub Runner
    pdf_filename = f"Diagnostic_Report_{os.urandom(4).hex()}.pdf"
    generate_pdf_artifact(pdf_filename, issue_title, report_content)
    
    # 3. Upload PDF to WordPress Media Endpoint
    pdf_url = upload_pdf_to_wp(pdf_filename)
    
    # 4. Create WordPress Post with Embed and Download Link
    wp_content = f"{report_content}\n\n<hr/><p><b>Download Official PDF Artifact:</b> <a href='{pdf_url}' target='_blank'>📥 Download Diagnostic PDF</a></p>"
    
    payload = {
        "title": f"Diagnostic: {issue_title}",
        "content": wp_content,
        "status": "publish"
    }
    
    res = requests.post(f"{WP_URL}/uesp_record", json=payload, auth=(WP_USER, WP_PASS))
    if res.status_code == 201:
        print(f"🚀 Resolution Post Published with PDF Link!")

if __name__ == "__main__":
    # Test issue execution
    input_issue = os.getenv("USER_ISSUE_INPUT", "Ergonomic friction and physical balance failure in tactical deployment frame.")
    process_issue_and_sync("Structural Equilibrium Analysis", input_issue)
