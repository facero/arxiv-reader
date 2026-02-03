#!/usr/bin/env python3
"""
Standalone script to regenerate the index.html page
"""
import json
import os
from datetime import datetime

REPORTS_DIR = "reports"
METADATA_FILE = os.path.join(REPORTS_DIR, "archive_metadata.json")

def load_metadata():
    """Loads archive metadata from JSON file."""
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def generate_index_page():
    """Generates the main archive index page with search and statistics."""
    metadata = load_metadata()
    
    if not metadata:
        print("No archive data found. Skipping index page generation.")
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
    print(f"Generating archive index page: {index_path}...")
    
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 ArXiv Paper Archive</h1>
            <div class="subtitle">High Energy Astrophysics - Monthly Reports</div>
            
            <div style="text-align: center; margin-bottom: 20px;">
                <button onclick="openReadingList()" style="background: #e74c3c; color: white; padding: 12px 24px; border-radius: 5px; border: none; cursor: pointer; text-decoration: none; font-weight: 500; font-size: 1em; display: inline-block;">
                    📖 View Reading List (<span id="readingListCount">0</span>)
                </button>
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
        
        <!-- Hidden sync iframe for cross-file origin sync -->
        <iframe id="syncIframe" src="reading_list.html" style="display:none;"></iframe>

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
            
            // --- Master Origin Logic ---
            const MASTER_PAGE = 'reading_list.html';
            const MASTER_WINDOW_NAME = '_arxiv_reading_list';

            function openReadingList() {{
                window.open(MASTER_PAGE, MASTER_WINDOW_NAME);
            }}

            function getReadingList() {{
                const list = localStorage.getItem('arxivReadingList');
                return list ? JSON.parse(list) : [];
            }}
            
            function updateReadingListCount() {{
                const count = getReadingList().length;
                const countElement = document.getElementById('readingListCount');
                if (countElement) {{
                    countElement.textContent = count;
                }}
            }}

            // Listen for sync from Master Page
            window.addEventListener('message', function(event) {{
                const data = event.data;
                if (data.type === 'arxiv_reading_list_sync_full') {{
                    localStorage.setItem('arxivReadingList', JSON.stringify(data.list));
                    updateReadingListCount();
                }}
            }}, false);
            
            // Initialize on page load
            document.addEventListener('DOMContentLoaded', updateReadingListCount);
        </script>
    </body>
    </html>
    """
    
    with open(index_path, "w") as f:
        f.write(html_content)
    
    print(f"Archive index page saved to {os.path.abspath(index_path)}")

if __name__ == "__main__":
    generate_index_page()
