import requests
import re
import numpy as np
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib.request
import xml.etree.ElementTree as ET
import sys
import os
import argparse
import json
from datetime import datetime

# Import standalone generators
from regenerate_index import generate_index_page
from generate_reading_list import generate_reading_list_page

# Force unbuffered output
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass  # Fallback for older Python versions or weird environments

# --- LLM Configuration ---
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
CHAT_MODEL = "mistralai/ministral-3-3b"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

# --- Input Files ---
ARXIV_BASE_URL = "https://arxiv.org/list/astro-ph.HE"
BIBLIOGRAPHY_FILE = "my-bibtex-Feb-2026.bib"
PERSONA_FILE = "research_persona.txt"
IGNORED_KEYWORDS_FILE = "ignored-keywords.txt"

# --- Output Files ---
TOP_K_CANDIDATES = 25  # Number of top papers to re-score with LLM per month
REPORTS_DIR = "reports"
METADATA_FILE = os.path.join(REPORTS_DIR, "archive_metadata.json")

# --- Helper Logger ---
def log(msg):
    """
    Prints a message to console and appends it to the scraper.log file.

    Args:
        msg: The message string to log.
    """
    print(msg, flush=True)
    sys.stdout.flush()  # Force immediate output
    with open("scraper.log", "a") as f:
        f.write(msg + "\n")

# --- Keyword Filtering ---
def load_ignored_keywords(filepath):
    """
    Loads ignored keywords from a text file (one per line).

    Args:
        filepath: The absolute path to the text file containing keywords.

    Returns:
        A list of lowercase keywords extracted from the file.
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

    Args:
        papers: A list of paper dictionaries, each containing at least a 'title'.
        ignored_keywords: A list of lowercase keywords to filter against.

    Returns:
        The filtered list of paper dictionaries.
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

    Args:
        filepath: The absolute path to the .bib file.

    Returns:
        A list of unique titles extracted from the BibTeX entries.
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

    Args:
        url: The arXiv URL to scrape (e.g., a specific month's listing).

    Returns:
        A list of dictionaries, each containing 'title', 'link', and 'id'.
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

# --- 3. ArXiv API Enrichment ---
def enrich_papers_with_api(papers, batch_size=100):
    """
    Fetches authors (first 5) and summary for each paper using the ArXiv API.
    Updates the dictionaries in-place.

    Args:
        papers: A list of paper dictionaries to enrich.
        batch_size: The number of IDs to request per API call (default: 100).
    """
    if not papers:
        return

    log(f"Enriching {len(papers)} papers via ArXiv API...")
    
    # Create a map for quick access (strip 'arXiv:' prefix if present)
    # Filter out invalid IDs
    id_map = {}
    valid_papers = []
    
    for p in papers:
        if 'id' in p and p['id'] != 'N/A':
            clean_id = p['id'].replace('arXiv:', '').strip()
            # Handle cases where ID might have version suffix already or other noise? 
            # Usually from monthly list it's clean.
            id_map[clean_id] = p
            valid_papers.append(clean_id)
            
    if not valid_papers:
        log("No valid ArXiv IDs found for enrichment.")
        return

    num_batches = (len(valid_papers) + batch_size - 1) // batch_size
    
    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    
    for i in range(0, len(valid_papers), batch_size):
        batch_ids = valid_papers[i:i + batch_size]
        id_list_str = ",".join(batch_ids)
        api_url = f"http://export.arxiv.org/api/query?id_list={id_list_str}&start=0&max_results={len(batch_ids)}"
        
        log(f"  Fetching API batch {i//batch_size + 1}/{num_batches}...")
        
        try:
            with urllib.request.urlopen(api_url) as response:
                data = response.read()
                root = ET.fromstring(data)
                
                for entry in root.findall('atom:entry', namespaces):
                    entry_id_url = entry.find('atom:id', namespaces).text
                    # Extract ID from URL (http://arxiv.org/abs/1234.5678v1)
                    # We strip the version (v1, v2) to match our list
                    full_id = entry_id_url.split('/abs/')[-1]
                    versionless_id = full_id.split('v')[0]
                    
                    target_paper = None
                    if versionless_id in id_map:
                        target_paper = id_map[versionless_id]
                    elif full_id in id_map: # Just in case
                        target_paper = id_map[full_id]
                        
                    if target_paper:
                        # Get Summary
                        summary_elem = entry.find('atom:summary', namespaces)
                        if summary_elem is not None and summary_elem.text:
                            # Clean up summary (remove newlines, extra spaces)
                            target_paper['summary'] = " ".join(summary_elem.text.strip().split())
                        else:
                            target_paper['summary'] = ""
                            
                        # Get Authors (first 5)
                        auth_elements = entry.findall('atom:author/atom:name', namespaces)
                        target_paper['authors'] = [a.text for a in auth_elements[:5]]
                        
        except Exception as e:
            log(f"  Error processing batch: {e}")
            # Do not crash; continue to next batch or proceed with missing metadata
            
    log("Enrichment complete.")

# --- 4. LMStudio Interaction ---
def get_embedding(text):
    """
    Fetches the embedding vector for a single string using the LMStudio/OpenAI-compatible API.

    Args:
        text: The text string to embed.

    Returns:
        A list of floats representing the embedding vector, 
        or None if the request fails.
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
    """
    Generates a chat completion using the LMStudio/OpenAI-compatible API.

    Args:
        messages: A list of message objects (role and content).
        max_tokens: The maximum number of tokens to generate (default: 500).

    Returns:
        The generated response content string, or None if the request fails.
    """
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

    Args:
        bib_titles: A list of publication titles from the user's bibliography.

    Returns:
        A one-paragraph summary of research interests, or None if generation fails.
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
    Hybrid scoring mechanism for paper relevance:
    1. Embedding similarity (fast vector search filter)
    2. LLM Persona-based Re-ranking (deep semantic analysis)

    Args:
        user_bib: The user's full bibliography (titles).
        arxiv_papers: The list of new arXiv papers to score.

    Returns:
        A sorted list of the most relevant papers with attached scores.
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
    
    # Enrich only the top candidates with API data (Summary, Authors)
    # This saves massive amounts of API calls
    enrich_papers_with_api(top_candidates)
    
    # 5. LLM Re-Scoring
    user_persona = generate_user_persona(user_bib)
    
    log("Scoring Top Candidates with LLM...")
    final_results = []
    
    for i, paper in enumerate(top_candidates):
        log(f"Scoring candidate {i+1}/{len(top_candidates)}: {paper['title'][:50]}...")
        summary_text = paper.get('summary', '')
        # Only use first few authors for prompt if list is long
        authors_list = paper.get('authors', [])
        authors_str = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
        
        prompt = f"""
        Does the following paper match this research persona?
        
        Persona: {user_persona}
        
        Paper Title: "{paper['title']}"
        Authors: {authors_str}
        Paper Summary: "{summary_text}"
        
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

# --- Main Logic ---
def generate_html_report(results, month_year):
    """
    Generates a stylized HTML report for a specific month's scored papers.

    Args:
        results: A list of paper dictionaries with scores and metadata.
        month_year: The month identifier in YYYY-MM format.
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
        
        # Prepare authors and summary
        authors_list = p.get('authors', [])
        authors_str = ", ".join(authors_list)
        summary_text = p.get('summary', '')
        
        html_content += f"""
        <div class="paper" style="border-left-color: {color}" id="paper-{paper_id}">
            <div class="header">
                <h2 class="title"><a href="{p['link']}" target="_blank">{i+1}. {p['title']}</a></h2>
                <div class="score-badge" style="background-color: {color}" title="Final Score = (Vector Similarity × 100 + LLM Score) / 2 = ({vector_s:.2f} × 100 + {llm_s}) / 2 = {score:.1f}">{score:.1f}</div>
            </div>
            
            <div style="margin-bottom: 10px; color: #555; font-style: italic;">
                {authors_str}
            </div>
            
            <details style="margin-bottom: 15px;">
                <summary style="cursor: pointer; color: #3498db; font-weight: 500;">Abstract</summary>
                <p style="margin-top: 5px; line-height: 1.5; color: #444;">{summary_text}</p>
            </details>
            
            <div class="meta">
                <div class="details">
                    <span class="metric" title="Cosine Similarity with Bibliography">Vector Similarity: <b>{vector_s:.2f}</b></span>
                    <span class="metric" title="LLM Relevance Score (0-100)">LLM Persona Score: <b>{llm_s}</b></span>
                    <span class="metric"><a href="{p['link']}" target="_blank">View on ArXiv</a></span>
                    <button class="add-to-reading-list" onclick="addToReadingList('{paper_id}', '{escaped_title}', '{p['link']}', {score:.1f}, '{month_year}', `{summary_text}`)" title="Add to Reading List">
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
    
    <!-- Hidden sync iframe for cross-file origin sync -->
    <iframe id="syncIframe" src="reading_list.html" style="display:none;"></iframe>
    
    <script>
        // --- Synchronization Logic (Master Origin: reading_list.html) ---
        const MASTER_PAGE = 'reading_list.html';
        const MASTER_WINDOW_NAME = '_arxiv_reading_list';
        
        function getReadingList() {
            const list = localStorage.getItem('arxivReadingList');
            return list ? JSON.parse(list) : [];
        }
        
        function saveReadingList(list) {
            localStorage.setItem('arxivReadingList', JSON.stringify(list));
            updateReadingListCount();
            markAddedPapers();
        }
        
        function addToReadingList(id, title, link, score, month, summary) {
            // 1. Construct sync URL for Master Page
            const params = new URLSearchParams();
            params.set('add_id', id);
            params.set('title', title);
            params.set('link', link);
            params.set('score', score);
            params.set('month', month);
            params.set('summary', summary);
            
            const syncUrl = `${MASTER_PAGE}?${params.toString()}`;
            
            // 2. Open/Focus Master Page in named window
            const masterWin = window.open(syncUrl, MASTER_WINDOW_NAME);
            
            // 3. Fallback: Save locally if Master Page blocked or etc.
            // (Standard behavior remains for local feedback)
            let localList = getReadingList();
            if (!localList.some(item => item.id === id)) {
                localList.push({ id, title, link, score, month, summary, addedDate: new Date().toISOString() });
                saveReadingList(localList);
            }
            
            showToast('Adding to Reading List... 📖');
            if (masterWin) masterWin.blur(); // Bring focus back to report (if possible)
            window.focus();
        }

        // Listen for sync messages from Master Page
        window.addEventListener('message', function(event) {
            // Security: In local file context, origin might be 'null'
            const data = event.data;
            
            if (data.type === 'arxiv_reading_list_synced') {
                showToast('Synced with Master List! ✅');
            } else if (data.type === 'arxiv_reading_list_sync_full') {
                // Received full list from hidden iframe on load
                console.log('Master list sync received:', data.list);
                saveReadingList(data.list); // Update local cache
            }
        }, false);
        
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
        
        function markAddedPapers() {
            const readingList = getReadingList();
            // Reset all buttons first (optional but safer)
            document.querySelectorAll('.add-to-reading-list').forEach(btn => {
                btn.classList.remove('added');
                btn.textContent = '📖 Add to List';
            });
            
            readingList.forEach(item => {
                const paperCard = document.getElementById(`paper-${item.id}`);
                if (paperCard) {
                    const buttons = paperCard.querySelectorAll('.add-to-reading-list');
                    buttons.forEach(btn => {
                        btn.classList.add('added');
                        btn.textContent = '✓ Added';
                    });
                }
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
    """
    Returns the current month in YYYY-MM format.

    Returns:
        Current system date formatted as 'YYYY-MM'.
    """
    return datetime.now().strftime("%Y-%m")

def load_metadata():
    """
    Loads archive metadata from the central JSON file.

    Returns:
        A dictionary containing metadata for all processed months.
    """
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        log(f"Error loading metadata: {e}")
        return {}

def save_metadata(metadata):
    """
    Saves the archive metadata to a JSON file.

    Args:
        metadata: The dictionary of metadata to persist.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        log(f"Error saving metadata: {e}")

def update_archive_metadata(month_year, total_papers, filtered_papers, top_matches, excluded_count):
    """
    Updates or creates the metadata entry for a specific month.

    Args:
        month_year: The month identifier in YYYY-MM format.
        total_papers: Total papers fetched from arXiv initially.
        filtered_papers: Papers remaining after keyword filtering.
        top_matches: Number of top papers scored and reported.
        excluded_count: Number of papers removed by keyword filters.
    """
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

# Note: generate_index_page and generate_reading_list_page are now imported from standalone scripts.
# The internal definitions have been removed to ensure a single source of truth.





# --- Main Workflow ---
def process_month(month_year=None):
    """
    The master workflow function to process arXiv papers for a specific month.
    Handles fetching, filtering, scoring, report generation, and index updates.

    Args:
        month_year: Month in YYYY-MM format. If None, defaults to current month.
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
    arxiv_url = f"{ARXIV_BASE_URL}/{month_year}?skip=0&show=2000"
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
    
    log(f"\n Successfully processed {month_year}")
    log(f"   Total papers: {total_papers}")
    log(f"   After filtering: {filtered_papers}")
    log(f"   Top matches: {len(results)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArXiv Paper Scorer - Monthly Archive System")
    parser.add_argument(
        "--month",
        type=str,
        nargs='*',  # Accept zero or more arguments
        help="Month(s) to process in YYYY-MM format (e.g. 2025-08 2025-09). Default: current month.",
        default=[]
    )
    
    args = parser.parse_args()
    
    months_to_process = args.month
    
    # If no months provided, default to current month
    if not months_to_process:
        months_to_process = [None]
        
    log(f"Starting batch processing for months: {months_to_process if months_to_process != [None] else '[Current Month]'}")
    
    for m in months_to_process:
        try:
            process_month(month_year=m)
        except Exception as e:
            log(f"Error processing month {m}: {e}")
            # Continue to next month if one fails


