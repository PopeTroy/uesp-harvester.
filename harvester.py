import os
import requests

# GitHub Secrets automatically fill these variables
WP_URL = "https://celsiustechmediagroup.co.za/wp-json/wp/v2/uesp_record"
WP_USER = os.getenv("WP_USERNAME")
WP_PASS = os.getenv("WP_APP_PASSWORD")
NEWS_KEY = os.getenv("NEWS_API_KEY")

def sync_to_wp(title, content):
    payload = {"title": title, "content": content, "status": "publish"}
    res = requests.post(WP_URL, json=payload, auth=(WP_USER, WP_PASS))
    if res.status_code == 201:
        print(f"✅ Success: {title}")
    else:
        print(f"❌ Failed: {res.status_code} - {res.text}")

def run_harvest():
    # 1. Fetch Google News (South Africa Construction)
    news_res = requests.get(f"https://newsapi.org/v2/everything?q=South+Africa+Construction&apiKey={NEWS_KEY}").json()
    if news_res.get('articles'):
        for art in news_res['articles'][:2]:
            sync_to_wp(f"News: {art['title']}", art['url'])

    # 2. Fetch Academic Journals (OpenAlex - Physics/Tech)
    acad_res = requests.get("https://api.openalex.org/works?search=quantum+construction&filter=is_oa:true").json()
    if acad_res.get('results'):
        for paper in acad_res['results'][:2]:
            sync_to_wp(f"Academic: {paper['display_name']}", paper['doi'])

if __name__ == "__main__":
    run_harvest()
