import os
import requests

# Secrets & Configuration
WP_URL = "https://celsiustechmediagroup.co.za/wp-json/wp/v2/uesp_record"
WP_USER = os.getenv("WP_USERNAME")
WP_PASS = os.getenv("WP_APP_PASSWORD")
NEWS_KEY = os.getenv("NEWS_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
# Using NVIDIA Nemotron / Llama-3-70B NIM endpoint
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"

# PhD Level System Architecture Prompt
SYSTEM_PROMPT = (
    "You are an elite PhD-level AI Research Agent operating at the convergence of Advanced Quantum Physics, "
    "Astrophysics, Physical Ergonomics, and Design Ergonomics. You analyze domain inputs using strategic Shinobi "
    "tactical frameworks (infiltrative diagnostic execution, ocular precision analysis, chakra/energy flow equilibrium). "
    "Generate a structured, rigorous Diagnostic Report evaluating design ergonomics, structural integrity, "
    "and systemic communication protocols."
)

def query_nvidia_nim(prompt_text):
    """Sends a payload to NVIDIA NIM API for advanced reasoning."""
    if not NVIDIA_API_KEY:
        print("❌ Error: NVIDIA_API_KEY environment variable missing.")
        return prompt_text

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    try:
        response = requests.post(NVIDIA_ENDPOINT, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"⚠️ NVIDIA API Warning: {response.status_code} - {response.text}")
            return prompt_text
    except Exception as e:
        print(f"❌ NVIDIA API Exception: {e}")
        return prompt_text

def sync_to_wp(title, raw_content):
    """Enhances content via NVIDIA agent and posts to WordPress endpoint."""
    print(f"⚡ Processing through NVIDIA NIM Agent Engine: {title}")
    
    analysis_prompt = (
        f"Subject Title: {title}\n"
        f"Input Source Data: {raw_content}\n\n"
        "Generate a diagnostic resolution assessing ergonomics, physical forces, and optimized strategic deployment."
    )
    
    enhanced_content = query_nvidia_nim(analysis_prompt)
    
    payload = {
        "title": title,
        "content": enhanced_content,
        "status": "publish"
    }
    
    res = requests.post(WP_URL, json=payload, auth=(WP_USER, WP_PASS))
    if res.status_code == 201:
        print(f"✅ Success: Published '{title}'")
    else:
        print(f"❌ Sync Failed: {res.status_code} - {res.text}")

def run_harvest():
    # 1. Fetch Google News (South Africa Construction)
    if NEWS_KEY:
        news_url = f"https://newsapi.org/v2/everything?q=South+Africa+Construction&apiKey={NEWS_KEY}"
        try:
            news_res = requests.get(news_url).json()
            if news_res.get('articles'):
                for art in news_res['articles'][:2]:
                    sync_to_wp(f"News Analysis: {art['title']}", f"URL: {art['url']}\nSummary: {art.get('description', '')}")
        except Exception as e:
            print(f"❌ News API Error: {e}")

    # 2. Fetch Academic Journals (OpenAlex - Physics/Tech)
    try:
        acad_res = requests.get("https://api.openalex.org/works?search=quantum+construction&filter=is_oa:true").json()
        if acad_res.get('results'):
            for paper in acad_res['results'][:2]:
                doi = paper.get('doi', 'N/A')
                title = paper.get('display_name', 'Academic Paper')
                sync_to_wp(f"Academic Audit: {title}", f"DOI: {doi}")
    except Exception as e:
        print(f"❌ OpenAlex API Error: {e}")

if __name__ == "__main__":
    run_harvest()
