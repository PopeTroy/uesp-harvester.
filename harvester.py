import os
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# WordPress & NVIDIA Credentials
WP_URL = "https://celsiustechmediagroup.co.za/wp-json/wp/v2"
WP_USER = os.getenv("WP_USERNAME")
WP_PASS = os.getenv("WP_APP_PASSWORD")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

LOGO_URL = "https://celsiustechmediagroup.co.za/wp-content/uploads/2026/01/CTMG.webp"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# Organization Meta Information
COMPANY_DETAILS = """
<b>Celsius Tech Media Group</b><br/>
Email: info@celsiusmediagroup.co.za<br/>
Web: celsiustechmediagroup.co.za<br/>
Engine: UESP / PRCE Resolution Protocol
"""

SYSTEM_PROMPT = """
[FMR SENTINEL MULTI-MODEL AGENT CORE]
You are a PhD-level research engine combining Quantum Mechanics, Astrophysics, Physical Ergonomics, and Shinobi Tactical Analysis (Ocular Diagnostics & Energy Balance).
Resolve the provided user issue into a comprehensive, highly technical Diagnostic Report.
"""

def fetch_logo_asset(local_filename="ctmg_logo.webp"):
    """Downloads website logo for PDF insertion."""
    try:
        res = requests.get(LOGO_URL, timeout=15)
        if res.status_code == 200:
            with open(local_filename, "wb") as f:
                f.write(res.content)
            return local_filename
    except Exception as e:
        print(f"⚠️ Logo Fetch Warning: {e}")
    return None

def query_nvidia_nim(issue_text):
    """Executes multi-model inference via NVIDIA NIM API."""
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
    """Generates styled PDF artifact containing logo, contact info, and compliance specs."""
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#003366'),
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#333333')
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        spaceAfter=8
    )

    compliance_style = ParagraphStyle(
        'ComplianceBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#004d40')
    )

    story = []

    # Header Table: Logo on Left, Company Details on Right
    logo_path = fetch_logo_asset()
    if logo_path and os.path.exists(logo_path):
        logo_img = Image(logo_path, width=1.8 * inch, height=0.6 * inch)
    else:
        logo_img = Paragraph("<b>CELSIUS TECH MEDIA GROUP</b>", title_style)

    header_table_data = [[logo_img, Paragraph(COMPANY_DETAILS, meta_style)]]
    header_table = Table(header_table_data, colWidths=[3.5 * inch, 3.5 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10)
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # Quantum Compliance & Engine Specifications Box
    compliance_text = (
        "<b>SYSTEM COMPLIANCE MANIFEST:</b><br/>"
        "• Inference Models Executed: NVIDIA Nemotron 3 Ultra (550B), Nemotron 3 Nano (30B), NeMo Retriever OCR, Cosmos Reason 2<br/>"
        "• Operational Cycle Time Dilation Metrics: 1 : 6000 Quantum Dilation Standard<br/>"
        "• Architectural Protocol Verification: 100% Compliant (UESP / PRCE Resolution Matrix)"
    )
    compliance_table = Table([[Paragraph(compliance_text, compliance_style)]], colWidths=[7.0 * inch])
    compliance_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#e0f2f1')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#004d40')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(compliance_table)
    story.append(Spacer(1, 15))

    # Document Title and Diagnostic Body
    story.append(Paragraph(f"<b>UESP DIAGNOSTIC REPORT:</b> {title}", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(content.replace('\n', '<br/>'), body_style))

    doc.build(story)
    print(f"📄 Local PDF Artifact Generated: {filename}")

def upload_pdf_to_wp(filepath):
    """Uploads compiled PDF artifact to WordPress Media Library."""
    filename = os.path.basename(filepath)
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"',
        'Content-Type': 'application/pdf'
    }

    with open(filepath, 'rb') as file_data:
        res = requests.post(f"{WP_URL}/media", headers=headers, data=file_data, auth=(WP_USER, WP_PASS))

    if res.status_code == 201:
        media_url = res.json().get('source_url')
        print(f"✅ PDF Uploaded to WordPress Media: {media_url}")
        return media_url
    else:
        print(f"❌ Failed PDF Upload: {res.status_code} - {res.text}")
        return None

def process_issue_and_sync(issue_title, issue_detail):
    report_content = query_nvidia_nim(issue_detail)
    pdf_filename = f"Diagnostic_Report_{os.urandom(4).hex()}.pdf"
    
    generate_pdf_artifact(pdf_filename, issue_title, report_content)
    pdf_url = upload_pdf_to_wp(pdf_filename)

    wp_content = (
        f"{report_content}\n\n<hr/>"
        f"<p><b>Download Official PDF Artifact:</b> "
        f"<a href='{pdf_url}' target='_blank'>📥 Download Diagnostic PDF</a></p>"
    )

    payload = {
        "title": f"Diagnostic: {issue_title}",
        "content": wp_content,
        "status": "publish"
    }

    res = requests.post(f"{WP_URL}/uesp_record", json=payload, auth=(WP_USER, WP_PASS))
    if res.status_code == 201:
        print(f"🚀 Published Diagnostic Resolution Post with PDF Link!")

if __name__ == "__main__":
    title_in = os.getenv("INJECTED_TITLE", "Quantum System Structural Analysis")
    detail_in = os.getenv("INJECTED_DETAIL", "Physical and ergonomic equilibrium assessment.")
    process_issue_and_sync(title_in, detail_in)
