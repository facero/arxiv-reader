import requests
import re
import numpy as np
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys
import os
import argparse
import json
from datetime import datetime

# --- Configuration ---
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
CHAT_MODEL = "mistralai/ministral-3-3b"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
BIBLIOGRAPHY_FILE = "my-bibtex-Feb-2026.bib"
PERSONA_FILE = "research_persona.txt"
IGNORED_KEYWORDS_FILE = "ignored-keywords.txt"
TOP_K_CANDIDATES = 30  # Number of top papers to re-score with LLM
REPORTS_DIR = "reports"
METADATA_FILE = os.path.join(REPORTS_DIR, ".archive_metadata.json")

# --- Helper Logger ---
def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()  # Force immediate output
    with open("scraper.log", "a") as f:
        f.write(msg + "\n")

# --- Keyword Filtering ---
def load_ignored_keywords(filepath):
    """
    Loads ignored keywords from a text file (one per line).
    Returns a list of lowercase keywords.
    """
    if not os.path.exists(filepath):
        log(f"Warning: {filepath} not found. No keywords will be filtered.")
        return []
    
    try:
        with open(filepath, 'r') as f:
            keywords = [line.strip().lower() for line in f if line.strip()]
        log(f"Loaded {len(keywords)} ignored keywords from {filepath}")
        return keywords
    except Exception as e:
        log(f"Error loading ignored keywords: {e}")
        return []

def filter_by_keywords(papers, ignored_keywords):
    """
    Filters out papers whose titles contain any of the ignored keywords (case-insensitive).
    """
    if not ignored_keywords:
        return papers
    
    filtered = []
    excluded_count = 0
    
    for paper in papers:
        title_lower = paper['title'].lower()
        # Check if any ignored keyword appears in the title
        if any(keyword in title_lower for keyword in ignored_keywords):
            excluded_count += 1
            continue
        filtered.append(paper)
    
    log(f"Filtered out {excluded_count} papers based on ignored keywords. {len(filtered)} papers remaining.")
    return filtered


# --- 1. Bibliography Parsing ---
def load_bibliography(filepath):
    """
    Parses a BibTeX file using regex to extract titles.
    Avoids `bibtexparser` dependency due to installation issues.
    """
    log(f"Loading bibliography from {filepath} (Regex Mode)...")
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Regex to find title = { ... } or title = " ... "
        # We assume standard formatting. 
        # Multi-line titles in bibtex often have whitespace/newlines.
        # This regex looks for 'title', optional whitespace, '=', optional whitespace, 
        # then '{' or '"', then capture until the matching closing brace/quote.
        # This is basic and might miss some complex nested braces, but good enough for ADS exports.
        
        titles = []
        
        # Pattern 1: title = { ... }
        # Non-greedy match until the first closing brace. 
        # NOTE: Nested braces will break this. ADS usually doesn't nest heavily in titles unless math.
        matches_braces = re.findall(r'title\s*=\s*\{(.+?)\},', content, re.IGNORECASE | re.DOTALL)
        titles.extend(matches_braces)

        # Pattern 2: title = " ... "
        matches_quotes = re.findall(r'title\s*=\s*\"(.+?)\",', content, re.IGNORECASE | re.DOTALL)
        titles.extend(matches_quotes)
        
        clean_titles = []
        for t in titles:
            # Clean up newlines and extra spaces
            ct = t.replace('\n', ' ').strip()
            # Remove inner braces often found in BibTeX like {Fermi}
            ct = ct.replace('{', '').replace('}', '')
            if ct:
                clean_titles.append(ct)
        
        # Deduplicate
        clean_titles = list(set(clean_titles))
        
        log(f"Found {len(clean_titles)} unique entries in bibliography.")
        return clean_titles
    except Exception as e:
        log(f"Error loading bibliography: {e}")
        return []

# --- 2. ArXiv Scraping ---
def fetch_arxiv_postings(url):
    """
    Scrapes the arXiv monthly list for titles and links.
    """
    log(f"Fetching arXiv listings from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        papers = []
        
        # ArXiv lists usually use <dl> with <dt> (id/link) and <dd> (meta)
        # But the specific monthly view might differ. 
        # Inspecting standard arXiv list format:
        # It's often pairs of <dt> and <dd>.
        
        dts = soup.find_all('dt')
        dds = soup.find_all('dd')
        
        if len(dts) != len(dds):
            log(f"Warning: Mismatch in dt/dd counts ({len(dts)} vs {len(dds)}). Parsing might be messy.")
        
        for dt, dd in zip(dts, dds):
            # Extract Title
            title_div = dd.find('div', class_='list-title')
            if title_div:
                # Remove "Title: " prefix
                title_text = title_div.get_text(strip=True).replace('Title:', '').strip()
                
                # Extract ID/Link
                # Structure is <dt><a title="Abstract" href="...">...</a></dt>
                link_anchor = dt.find('a', title='Abstract')
                link = urljoin(url, link_anchor['href']) if link_anchor else "N/A"
                arxiv_id = link_anchor.get_text(strip=True) if link_anchor else "N/A"

                papers.append({
                    'title': title_text,
                    'link': link,
                    'id': arxiv_id
                })
                
        log(f"Found {len(papers)} papers on arXiv page.")
        return papers

    except Exception as e:
        log(f"Error fetching arXiv data: {e}")
        return []

# --- 3. LMStudio Interaction ---
def get_embedding(text):
    """
    Fetches the embedding vector for a single string.
    """
    url = f"{LMSTUDIO_BASE_URL}/embeddings"
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]['embedding']
    except Exception as e:
        # Fail gracefully - return None for failed embeddings
        # log(f"Error getting embedding: {e}")
        return None

def get_chat_completion(messages, max_tokens=500):
    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        log(f"Error in chat completion: {e}")
        return None

# --- 4. Logic Core ---

def generate_user_persona(bib_titles):
    """
    Summarizes the user's research interests based on their bibliography.
    Caches the result to a file to avoid re-running on every execution.
    """
    # 1. Check if persona file exists
    if os.path.exists(PERSONA_FILE):
        log(f"Loading existing Research Persona from {PERSONA_FILE}...")
        try:
            with open(PERSONA_FILE, 'r') as f:
                persona = f.read()
            if persona.strip():
                log("Persona loaded successfully.")
                return persona
        except Exception as e:
            log(f"Error reading persona file: {e}. Regenerating...")

    # 2. Regenerate if not found
    log("Generating User Research Persona (this may take 10-20 seconds)...")
    
    # We can't feed thousands of titles. Let's take a sample or the most recent ones if possible.
    # Assuming the list is roughly chronological or we just take a random sample of 50.
    sample_size = min(len(bib_titles), 50)
    # Just take the first 50 for now (often bib files are sorted by year desc)
    sample_titles = bib_titles[:sample_size]
    
    titles_block = "\n".join([f"- {t}" for t in sample_titles])
    
    system_prompt = "You are an expert research scientist in Astrophysics."
    user_prompt = f"""
    Here is a list of publications from my bibliography:
    {titles_block}

    Based on these titles, write a concise "Research Interest Persona" for me. 
    Identify specific astrophysical objects (e.g. Magnetars, GRBs, AGNs), methods (e.g. spectroscopy, simulations), and energy regimes (e.g. X-ray, Gamma-ray).
    Keep it to one paragraph.
    """
    
    persona = get_chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    # 3. Save to file
    if persona:
        try:
            with open(PERSONA_FILE, 'w') as f:
                f.write(persona)
            log(f"Saved generated persona to {PERSONA_FILE}")
        except Exception as e:
            log(f"Error saving persona file: {e}")
    
    log(f"\n--- Generated Persona ---\n{persona}\n-------------------------\n")
    return persona

def score_papers_hybrid(user_bib, arxiv_papers):
    """
    Hybrid scoring:
    1. Embedding similarity (fast filter)
    2. LLM Persona-based Re-ranking (slow, high quality)
    """
    # 1. Embed user bibliography
    # To save time, let's sample the user's bib (e.g., last 100 papers)
    bib_sample = user_bib[:100]
    log(f"Computing embeddings for {len(bib_sample)} bibliography entries...")
    
    bib_embeddings = []
    for i, title in enumerate(bib_sample):
        emb = get_embedding(title)
        if emb:
            bib_embeddings.append(emb)
        
        if (i + 1) % 20 == 0 or (i + 1) == len(bib_sample):
            log(f"  Embedded {i+1}/{len(bib_sample)} bibliography entries...")
    
    if not bib_embeddings:
        log("No embeddings generated for bibliography. Exiting.")
        return []
        
    bib_matrix = np.array(bib_embeddings) # Shape: (N_bib, D)
    
    # 2. Embed ArXiv papers
    log(f"Computing embeddings for {len(arxiv_papers)} new ArXiv papers...")
    paper_embeddings = []
    valid_papers = [] # Keep track of papers that successfully embedded
    
    for i, paper in enumerate(arxiv_papers):
        emb = get_embedding(paper['title'])
        if emb:
            paper_embeddings.append(emb)
            valid_papers.append(paper)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(arxiv_papers):
            log(f"  Embedded {i+1}/{len(arxiv_papers)} papers...")
            
    if not paper_embeddings:
        log("No embeddings for ArXiv papers.")
        return []

    paper_matrix = np.array(paper_embeddings) # Shape: (M_papers, D)
    
    # 3. Vector Similarity
    # We want to find papers that are close to ANY of the user's papers.
    # Score(paper_i) = max(cosine_sim(paper_i, bib_j)) for all j
    
    log("Computing vector similarity scores...")
    
    # Normalize vectors for cosine similarity
    bib_norm = bib_matrix / np.linalg.norm(bib_matrix, axis=1, keepdims=True)
    paper_norm = paper_matrix / np.linalg.norm(paper_matrix, axis=1, keepdims=True)
    
    # Similarity matrix: (M_papers, N_bib)
    similarity_matrix = np.dot(paper_norm, bib_norm.T)
    
    # Max similarity per paper
    max_scores = np.max(similarity_matrix, axis=1)
    
    # Attach preliminary scores
    for i, paper in enumerate(valid_papers):
        paper['vector_score'] = max_scores[i]
        
    # 4. Filter Top Candidates
    # Take top K for LLM re-scoring
    valid_papers.sort(key=lambda x: x['vector_score'], reverse=True)
    top_candidates = valid_papers[:TOP_K_CANDIDATES]
    
    log(f"\nTop {len(top_candidates)} candidates selected for LLM re-scoring based on vector similarity.")
    
    # 5. LLM Re-Scoring
    user_persona = generate_user_persona(user_bib)
    
    log("Scoring Top Candidates with LLM...")
    final_results = []
    
    for i, paper in enumerate(top_candidates):
        log(f"Scoring candidate {i+1}/{len(top_candidates)}: {paper['title'][:50]}...")
        prompt = f"""
        Does the following paper title match this research persona?
        
        Persona: {user_persona}
        
        Paper Title: "{paper['title']}"
        
        Answer with a single number from 0 to 100, where 100 is a perfect match and 0 is irrelevant. 
        Only provides the number.
        """
        
        score_str = get_chat_completion([
             {"role": "user", "content": prompt}
        ], max_tokens=10)
        
        try:
            # Extract number from string (handle potential extra text)
            match = re.search(r'\d+', score_str)
            llm_score = int(match.group()) if match else 0
        except:
            llm_score = 0
            
        paper['llm_score'] = llm_score
        # Combined score? Or just trust LLM? Let's average them (normalized)
        # Vector score is 0-1 (cosine). LLM is 0-100.
        paper['final_score'] = (paper['vector_score'] * 100 + llm_score) / 2
        final_results.append(paper)
        
    # Sort by final score
    final_results.sort(key=lambda x: x['final_score'], reverse=True)
    return final_results

# --- Main ---
def main():
    # 1. Load Bib
    bib_entries = load_bibliography(BIBLIOGRAPHY_FILE)
    if not bib_entries:
        print("Please ensure the .bib file is correct.")
        return

    # 2. Fetch ArXiv
def generate_html_report(results, month_year):
    """
    Generates a nice HTML report of the scored papers for a specific month.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = os.path.join(REPORTS_DIR, f"{month_year}.html")
    log(f"Generating HTML report: {filename}...")
    
    # Convert month_year to readable format
    try:
        month_obj = datetime.strptime(month_year, "%Y-%m")
        month_name = month_obj.strftime("%B %Y")
    except:
        month_name = month_year
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ArXiv Report - {month_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; color: #333; }}
            .nav {{ text-align: center; margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .nav a {{ margin: 0 15px; text-decoration: none; color: #3498db; font-weight: 500; }}
            .nav a:hover {{ text-decoration: underline; }}
            h1 {{ text-align: center; color: #2c3e50; margin-bottom: 5px; }}
            .subtitle {{ text-align: center; color: #666; margin-bottom: 20px; }}
            .stats {{ text-align: center; margin-bottom: 30px; padding: 15px; background: #ecf0f1; border-radius: 8px; }}
            .stats span {{ margin: 0 15px; font-size: 0.9em; }}
            .paper {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; border-left: 5px solid #3498db; transition: transform 0.2s; }}
            .paper:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }}
            .header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }}
            .score-badge {{ background-color: #3498db; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold; font-size: 1.1em; }}
            .title {{ font-size: 1.3em; margin: 0; }}
            .title a {{ text-decoration: none; color: #2c3e50; }}
            .title a:hover {{ color: #3498db; }}
            .meta {{ font-size: 0.9em; color: #7f8c8d; margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px; }}
            .details {{ display: flex; gap: 20px; }}
            .metric {{ background: #f0f2f5; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }}
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="index.html">📚 Archive Home</a>
            <a href="reading_list.html" style="background: #e74c3c; color: white; padding: 8px 15px; border-radius: 5px; margin-left: 10px;">📖 Reading List (<span id="readingListCount">0</span>)</a>
        </div>
        <h1>ArXiv Relevance Report</h1>
        <div class="subtitle">{month_name} - High Energy Astrophysics</div>
        <div class="stats">
            <span>📊 <b>{len(results)}</b> Top Matches</span>
        </div>
    """
    
    for i, p in enumerate(results):
        # Color coding for score
        score = p.get('final_score', 0)
        color = "#2ecc71" if score > 80 else "#f39c12" if score > 50 else "#e74c3c"
        
        vector_s = p.get('vector_score', 0)
        llm_s = p.get('llm_score', 0)
        
        # Create unique ID for this paper
        paper_id = p['link'].split('/')[-1]  # Extract arXiv ID from URL
        
        # Escape single quotes in title for JavaScript
        escaped_title = p['title'].replace("'", "\\'")
        
        html_content += f"""
        <div class="paper" style="border-left-color: {color}" id="paper-{paper_id}">
            <div class="header">
                <h2 class="title"><a href="{p['link']}" target="_blank">{i+1}. {p['title']}</a></h2>
                <div class="score-badge" style="background-color: {color}" title="Final Score = (Vector Similarity × 100 + LLM Score) / 2 = ({vector_s:.2f} × 100 + {llm_s}) / 2 = {score:.1f}">{score:.1f}</div>
            </div>
            <div class="meta">
                <div class="details">
                    <span class="metric" title="Cosine Similarity with Bibliography">Vector Similarity: <b>{vector_s:.2f}</b></span>
                    <span class="metric" title="LLM Relevance Score (0-100)">LLM Persona Score: <b>{llm_s}</b></span>
                    <span class="metric"><a href="{p['link']}" target="_blank">View on ArXiv</a></span>
                    <button class="add-to-reading-list" onclick="addToReadingList('{paper_id}', '{escaped_title}', '{p['link']}', {score:.1f}, '{month_year}')" title="Add to Reading List">
                        📖 Add to List
                    </button>
                </div>
            </div>
        </div>
        """
        
    html_content += """
    <style>
        .add-to-reading-list {
            background: #3498db;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background 0.3s;
        }
        .add-to-reading-list:hover {
            background: #2980b9;
        }
        .add-to-reading-list.added {
            background: #27ae60;
        }
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #27ae60;
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
        }
        .toast.show {
            opacity: 1;
        }
    </style>
    
    <div id="toast" class="toast"></div>
    
    <script>
        // Reading List Management
        function getReadingList() {
            const list = localStorage.getItem('arxivReadingList');
            return list ? JSON.parse(list) : [];
        }
        
        function saveReadingList(list) {
            localStorage.setItem('arxivReadingList', JSON.stringify(list));
            updateReadingListCount();
        }
        
        function addToReadingList(id, title, link, score, month) {
            const readingList = getReadingList();
            
            // Check if already in list
            if (readingList.some(item => item.id === id)) {
                showToast('Already in reading list!', '#f39c12');
                return;
            }
            
            // Add to list
            readingList.push({
                id: id,
                title: title,
                link: link,
                score: score,
                month: month,
                addedDate: new Date().toISOString()
            });
            
            saveReadingList(readingList);
            showToast('Added to reading list! 📖');
            
            // Update button appearance
            const buttons = document.querySelectorAll(`#paper-${id} .add-to-reading-list`);
            buttons.forEach(btn => {
                btn.classList.add('added');
                btn.textContent = '✓ Added';
            });
        }
        
        function updateReadingListCount() {
            const count = getReadingList().length;
            const countElement = document.getElementById('readingListCount');
            if (countElement) {
                countElement.textContent = count;
            }
        }
        
        function showToast(message, color = '#27ae60') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.style.background = color;
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 2000);
        }
        
        // Mark papers already in reading list
        function markAddedPapers() {
            const readingList = getReadingList();
            readingList.forEach(item => {
                const buttons = document.querySelectorAll(`#paper-${item.id} .add-to-reading-list`);
                buttons.forEach(btn => {
                    btn.classList.add('added');
                    btn.textContent = '✓ Added';
                });
            });
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            updateReadingListCount();
            markAddedPapers();
        });
    </script>
    </body>
    </html>
    """
    
    with open(filename, "w") as f:
        f.write(html_content)
    
    log(f"HTML report saved to {os.path.abspath(filename)}")


# --- Archive Management ---
def get_current_month():
    """Returns current month in YYYY-MM format."""
    return datetime.now().strftime("%Y-%m")

def load_metadata():
    """Loads archive metadata from JSON file."""
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        log(f"Error loading metadata: {e}")
        return {}

def save_metadata(metadata):
    """Saves archive metadata to JSON file."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        log(f"Error saving metadata: {e}")

def update_archive_metadata(month_year, total_papers, filtered_papers, top_matches, excluded_count):
    """Updates metadata for a specific month."""
    metadata = load_metadata()
    metadata[month_year] = {
        "processed_date": datetime.now().isoformat(),
        "total_papers": total_papers,
        "filtered_papers": filtered_papers,
        "top_matches": top_matches,
        "excluded_keywords": excluded_count
    }
    save_metadata(metadata)
    log(f"Updated archive metadata for {month_year}")

def generate_index_page():
    """Generates the main archive index page with search and statistics."""
    metadata = load_metadata()
    
    if not metadata:
        log("No archive data found. Skipping index page generation.")
        return
    
    # Sort months in reverse chronological order
    sorted_months = sorted(metadata.keys(), reverse=True)
    
    # Calculate statistics
    total_months = len(sorted_months)
    total_papers_all = sum(m['total_papers'] for m in metadata.values())
    avg_papers = total_papers_all / total_months if total_months > 0 else 0
    
    # Prepare data for Chart.js
    chart_labels = [m for m in reversed(sorted_months)]
    chart_data = [metadata[m]['total_papers'] for m in reversed(sorted_months)]
    
    index_path = os.path.join(REPORTS_DIR, "index.html")
    log(f"Generating archive index page: {index_path}...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ArXiv Paper Archive</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                background: white;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 2.5em;
            }}
            .subtitle {{
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }}
            .search-box {{
                margin-bottom: 30px;
                text-align: center;
            }}
            .search-box input {{
                width: 60%;
                padding: 12px 20px;
                font-size: 1em;
                border: 2px solid #3498db;
                border-radius: 25px;
                outline: none;
                transition: all 0.3s;
            }}
            .search-box input:focus {{
                border-color: #2980b9;
                box-shadow: 0 0 10px rgba(52, 152, 219, 0.3);
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-card .number {{
                font-size: 2.5em;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .stat-card .label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .chart-container {{
                margin-bottom: 30px;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
            }}
            .month-list {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 30px;
            }}
            .month-item {{
                background: white;
                border: 2px solid #ecf0f1;
                border-radius: 8px;
                padding: 15px;
                transition: all 0.3s;
                display: flex;
                flex-direction: column;
                align-items: stretch;
            }}
            .month-item:hover {{
                border-color: #3498db;
                transform: translateY(-3px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .month-item.hidden {{
                display: none;
            }}
            @media (max-width: 900px) {{
                .month-list {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}
            @media (max-width: 500px) {{
                .month-list {{
                    grid-template-columns: 1fr;
                }}
            }}
            .month-info {{
                flex: 1;
                margin-bottom: 10px;
            }}
            .month-title {{
                font-size: 1.1em;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
            }}
            .month-stats {{
                color: #7f8c8d;
                font-size: 0.8em;
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}
            .month-stats span {{
                display: block;
            }}
            .view-btn {{
                background: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                text-decoration: none;
                font-weight: 500;
                transition: background 0.3s;
                text-align: center;
                display: block;
            }}
            .view-btn:hover {{
                background: #2980b9;
            }}
    </head>
    <body>
        <div class="container">
            <h1>📚 ArXiv Paper Archive</h1>
            <div class="subtitle">High Energy Astrophysics - Monthly Reports</div>
            
            <div style="text-align: center; margin-bottom: 20px;">
                <a href="reading_list.html" style="background: #e74c3c; color: white; padding: 12px 24px; border-radius: 5px; text-decoration: none; font-weight: 500; display: inline-block;">
                    📖 View Reading List (<span id="readingListCount">0</span>)
                </a>
            </div>
            
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="🔍 Search by month or keyword..." onkeyup="filterMonths()">
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="number">{total_months}</div>
                    <div class="label">Months Archived</div>
                </div>
                <div class="stat-card">
                    <div class="number">{total_papers_all:,}</div>
                    <div class="label">Total Papers</div>
                </div>
                <div class="stat-card">
                    <div class="number">{avg_papers:.0f}</div>
                    <div class="label">Avg Papers/Month</div>
                </div>
            </div>
            
            <div class="month-list" id="monthList">
    """
    
    for month in sorted_months:
        data = metadata[month]
        try:
            month_obj = datetime.strptime(month, "%Y-%m")
            month_name = month_obj.strftime("%B %Y")
        except:
            month_name = month
        
        html_content += f"""
                <div class="month-item" data-month="{month}" data-name="{month_name.lower()}">
                    <div class="month-info">
                        <div class="month-title">📅 {month_name}</div>
                        <div class="month-stats">
                            <span>📊 {data['total_papers']} papers</span>
                            <span>✅ {data['top_matches']} matches</span>
                            <span>🚫 {data['excluded_keywords']} filtered</span>
                        </div>
                    </div>
                    <a href="{month}.html" class="view-btn">View Report →</a>
                </div>
        """
    
    html_content += f"""
            </div>
            
            <div class="chart-container">
                <canvas id="papersChart"></canvas>
            </div>
        </div>
        
        <script>
            // Chart.js configuration
            const ctx = document.getElementById('papersChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(chart_labels)},
                    datasets: [{{
                        label: 'Papers per Month',
                        data: {json.dumps(chart_data)},
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Papers Trend Over Time',
                            font: {{ size: 16 }}
                        }},
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Number of Papers'
                            }}
                        }}
                    }}
                }}
            }});
            
            // Search functionality
            function filterMonths() {{
                const input = document.getElementById('searchInput');
                const filter = input.value.toLowerCase();
                const monthItems = document.querySelectorAll('.month-item');
                
                monthItems.forEach(item => {{
                    const month = item.getAttribute('data-month');
                    const name = item.getAttribute('data-name');
                    if (month.includes(filter) || name.includes(filter)) {{
                        item.classList.remove('hidden');
                    }} else {{
                        item.classList.add('hidden');
                    }}
                }});
            }}
            
            // Reading List Counter
            function updateReadingListCount() {{
                const list = localStorage.getItem('arxivReadingList');
                const count = list ? JSON.parse(list).length : 0;
                const countElement = document.getElementById('readingListCount');
                if (countElement) {{
                    countElement.textContent = count;
                }}
            }}
            
            // Initialize on page load
            document.addEventListener('DOMContentLoaded', updateReadingListCount);
        </script>
    </body>
    </html>
    """
    
    with open(index_path, "w") as f:
        f.write(html_content)
    
    log(f"Archive index page saved to {os.path.abspath(index_path)}")

def generate_reading_list_page():
    """Generates the reading list page (client-side rendering from localStorage)."""
    reading_list_path = os.path.join(REPORTS_DIR, "reading_list.html")
    log(f"Generating reading list page: {reading_list_path}...")
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reading List - ArXiv Papers</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                text-align: center;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            .nav {
                text-align: center;
                margin-bottom: 20px;
            }
            .nav a {
                margin: 0 10px;
                text-decoration: none;
                color: #3498db;
                font-weight: 500;
            }
            .nav a:hover {
                text-decoration: underline;
            }
            .actions {
                text-align: center;
                margin-bottom: 30px;
            }
            .clear-btn {
                background: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 500;
            }
            .clear-btn:hover {
                background: #c0392b;
            }
            .paper {
                background: white;
                border: 2px solid #ecf0f1;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
                transition: all 0.3s;
                border-left: 5px solid #3498db;
            }
            .paper:hover {
                border-color: #3498db;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .paper-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 10px;
            }
            .paper-title {
                font-size: 1.2em;
                font-weight: 600;
                color: #2c3e50;
                flex: 1;
                margin-right: 15px;
            }
            .paper-title a {
                color: #2c3e50;
                text-decoration: none;
            }
            .paper-title a:hover {
                color: #3498db;
            }
            .score-badge {
                background: #3498db;
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-weight: bold;
                font-size: 0.9em;
            }
            .paper-meta {
                color: #7f8c8d;
                font-size: 0.9em;
                margin-bottom: 10px;
            }
            .paper-meta span {
                margin-right: 15px;
            }
            .remove-btn {
                background: #e74c3c;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.85em;
            }
            .remove-btn:hover {
                background: #c0392b;
            }
            .empty-state {
                text-align: center;
                padding: 60px 20px;
                color: #7f8c8d;
            }
            .empty-state h2 {
                color: #95a5a6;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav">
                <a href="index.html">📚 Archive Home</a>
            </div>
            
            <h1>📖 Reading List</h1>
            <div class="subtitle">Your Saved ArXiv Papers</div>
            
            <div class="actions">
                <button class="clear-btn" onclick="clearReadingList()">🗑️ Clear All</button>
            </div>
            
            <div id="readingListContainer"></div>
        </div>
        
        <script>
            function getReadingList() {
                const list = localStorage.getItem('arxivReadingList');
                return list ? JSON.parse(list) : [];
            }
            
            function saveReadingList(list) {
                localStorage.setItem('arxivReadingList', JSON.stringify(list));
            }
            
            function removeFromReadingList(id) {
                let readingList = getReadingList();
                readingList = readingList.filter(item => item.id !== id);
                saveReadingList(readingList);
                renderReadingList();
            }
            
            function clearReadingList() {
                if (confirm('Are you sure you want to clear your entire reading list?')) {
                    localStorage.removeItem('arxivReadingList');
                    renderReadingList();
                }
            }
            
            function renderReadingList() {
                const readingList = getReadingList();
                const container = document.getElementById('readingListContainer');
                
                if (readingList.length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <h2>📭 Your reading list is empty</h2>
                            <p>Add papers from the monthly reports to build your reading list!</p>
                            <p><a href="index.html" style="color: #3498db;">Browse Archive →</a></p>
                        </div>
                    `;
                    return;
                }
                
                // Sort by added date (newest first)
                readingList.sort((a, b) => new Date(b.addedDate) - new Date(a.addedDate));
                
                let html = '';
                readingList.forEach(paper => {
                    const addedDate = new Date(paper.addedDate).toLocaleDateString();
                    const scoreColor = paper.score > 80 ? '#2ecc71' : paper.score > 50 ? '#f39c12' : '#e74c3c';
                    
                    html += `
                        <div class="paper">
                            <div class="paper-header">
                                <div class="paper-title">
                                    <a href="${paper.link}" target="_blank">${paper.title}</a>
                                </div>
                                <div class="score-badge" style="background-color: ${scoreColor}">${paper.score.toFixed(1)}</div>
                            </div>
                            <div class="paper-meta">
                                <span>📅 ${paper.month}</span>
                                <span>🕒 Added: ${addedDate}</span>
                                <span><a href="${paper.link}" target="_blank">View on ArXiv →</a></span>
                            </div>
                            <button class="remove-btn" onclick="removeFromReadingList('${paper.id}')">Remove</button>
                        </div>
                    `;
                });
                
                container.innerHTML = html;
            }
            
            // Initialize on page load
            document.addEventListener('DOMContentLoaded', renderReadingList);
        </script>
    </body>
    </html>
    """
    
    with open(reading_list_path, "w") as f:
        f.write(html_content)
    
    log(f"Reading list page saved to {os.path.abspath(reading_list_path)}")




# --- Main ---
def main(month_year=None):
    """
    Main function to process arXiv papers for a specific month.
    
    Args:
        month_year: Month in YYYY-MM format. If None, uses current month.
    """
    # Default to current month if not specified
    if month_year is None:
        month_year = get_current_month()
    
    log(f"\n{'='*60}")
    log(f"Processing arXiv papers for: {month_year}")
    log(f"{'='*60}\n")
    
    # 1. Load Bib
    bib_entries = load_bibliography(BIBLIOGRAPHY_FILE)
    if not bib_entries:
        log("No bibliography entries found. Exiting.")
        return

    # 2. Fetch ArXiv
    arxiv_url = f"https://arxiv.org/list/astro-ph.HE/{month_year}?skip=0&show=2000"
    all_papers = fetch_arxiv_postings(arxiv_url)
    
    if not all_papers:
        log("No ArXiv papers found.")
        return
    
    total_papers = len(all_papers)
    
    # 2.5. Filter by ignored keywords
    ignored_keywords = load_ignored_keywords(IGNORED_KEYWORDS_FILE)
    filtered_papers_before = len(all_papers)
    all_papers = filter_by_keywords(all_papers, ignored_keywords)
    excluded_count = filtered_papers_before - len(all_papers)
    
    if not all_papers:
        log("No papers remaining after keyword filtering.")
        return
    
    filtered_papers = len(all_papers)
        
    # 3. Score
    results = score_papers_hybrid(bib_entries, all_papers)
    
    # 4. Output to console
    log("\n\n====== FINAL RESULTS ======\n")
    for i, p in enumerate(results):
        log(f"{i+1}. [Score: {p['final_score']:.1f}] {p['title']}")
        log(f"    Link: {p['link']}")
        log(f"    (Vector: {p['vector_score']:.2f}, LLM: {p['llm_score']})")
        log("-" * 40)
        
    # 5. Generate HTML report (monthly)
    generate_html_report(results, month_year)
    
    # 6. Update archive metadata
    update_archive_metadata(month_year, total_papers, filtered_papers, len(results), excluded_count)
    
    # 7. Regenerate index page
    generate_index_page()
    
    # 8. Generate reading list page
    generate_reading_list_page()
    
    log(f"\n✅ Successfully processed {month_year}")
    log(f"   Total papers: {total_papers}")
    log(f"   After filtering: {filtered_papers}")
    log(f"   Top matches: {len(results)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArXiv Paper Scorer - Monthly Archive System")
    parser.add_argument(
        "--month",
        type=str,
        help="Month to process in YYYY-MM format (default: current month)",
        default=None
    )
    
    args = parser.parse_args()
    main(month_year=args.month)

