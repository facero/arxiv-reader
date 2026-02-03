#!/usr/bin/env python3
"""
Standalone script to generate the reading_list.html page
"""
import os

REPORTS_DIR = "reports"

def generate_reading_list_page():
    """Generates the reading list page (client-side rendering from localStorage)."""
    reading_list_path = os.path.join(REPORTS_DIR, "reading_list.html")
    print(f"Generating reading list page: {reading_list_path}...")
    
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
            // --- Master Storage Logic ---
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

            // --- URL Parameter Synchronization ---
            function handleIncomingPaper() {
                const params = new URLSearchParams(window.location.search);
                const id = params.get('add_id');
                if (!id) return;

                const title = params.get('title');
                const link = params.get('link');
                const score = parseFloat(params.get('score'));
                const month = params.get('month');
                const summary = params.get('summary');

                let readingList = getReadingList();
                
                // Only add if not already present
                if (!readingList.some(item => item.id === id)) {
                    readingList.push({
                        id: id,
                        title: title,
                        link: link,
                        score: score,
                        month: month,
                        summary: summary,
                        addedDate: new Date().toISOString()
                    });
                    saveReadingList(readingList);
                }

                // Notify opener and clean URL
                if (window.opener) {
                    window.opener.postMessage({ type: 'arxiv_reading_list_synced', id: id }, '*');
                }
                
                // Remove params from URL without reloading
                const newUrl = window.location.pathname;
                window.history.replaceState({}, '', newUrl);
            }

            // --- Synchronization Probe (for monthly reports) ---
            function sendSyncProbe() {
                // If we are in an iframe or opened as a master origin
                if (window.parent !== window || window.opener) {
                    const readingList = getReadingList();
                    (window.parent || window.opener).postMessage({ 
                        type: 'arxiv_reading_list_sync_full', 
                        list: readingList 
                    }, '*');
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
                            
                            ${paper.summary ? `
                            <details style="margin-bottom: 15px;">
                                <summary style="cursor: pointer; color: #3498db; font-weight: 500;">Abstract</summary>
                                <p style="margin-top: 5px; line-height: 1.5; color: #444; font-size: 0.95em;">${paper.summary}</p>
                            </details>
                            ` : ''}

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
            document.addEventListener('DOMContentLoaded', () => {
                handleIncomingPaper();
                renderReadingList();
                sendSyncProbe(); // Notify parent on load
            });

            // Handle cases where the window stays open and a new paper is "added" via re-focus
            window.addEventListener('focus', () => {
                handleIncomingPaper();
                sendSyncProbe(); // Re-sync on focus/update
            });
        </script>
    </body>
    </html>
    """
    
    with open(reading_list_path, "w") as f:
        f.write(html_content)
    
    print(f"Reading list page saved to {os.path.abspath(reading_list_path)}")

if __name__ == "__main__":
    generate_reading_list_page()
