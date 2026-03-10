#!/bin/bash
cd /Users/xuyili/.openclaw/workspace

# 1. 抓取论文
python3 << 'PYEOF'
import sqlite3
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

DB = "/Users/xuyili/.openclaw/db/papers.db"
conn = sqlite3.connect(DB)
conn.execute("DELETE FROM papers")

topics = {
    'LLM': 'large language model',
    'CV': 'computer vision',  
    '多模态': 'multimodal',
    '新数据集': 'dataset benchmark',
    '模型压缩': 'model compression pruning',
    '综述': 'survey review',
    'RL': 'reinforcement learning'
}

def fetch_arxiv(query, topic):
    url = "http://export.arxiv.org/api/query"
    try:
        resp = requests.get(url, params={'search_query': f'all:{query}', 'max_results': 15, 'sortBy': 'submittedDate', 'sortOrder': 'descending'}, timeout=20)
        root = ET.fromstring(resp.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        for entry in root.findall('atom:entry', ns)[:15]:
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            abstract = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            link = entry.find('atom:id', ns).text
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            papers.append({'title': title, 'abstract': abstract, 'authors': ', '.join(authors[:3]), 'url': link, 'topic': topic})
        return papers
    except:
        return []

for topic, query in topics.items():
    papers = fetch_arxiv(query, topic)
    for p in papers:
        conn.execute("""INSERT INTO papers (title, authors, abstract, venue, year, url, topic, rating, fetched_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (p['title'], p['authors'], p['abstract'], 'arXiv', 2026, p['url'], p['topic'], 0, datetime.now().isoformat()))

conn.commit()
conn.close()
print("✅ 论文抓取完成")
PYEOF

# 2. 生成HTML（使用真正翻译）
python3 << 'PYEOF'
import sqlite3
import re
import time
from datetime import datetime
from deep_translator import GoogleTranslator

DB = "/Users/xuyili/.openclaw/db/papers.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
date = datetime.now().strftime('%Y-%m-%d')

# 初始化翻译器
translator = GoogleTranslator(source='en', target='zh-CN')

# 不翻译的专有名词
NO_TRANSLATE = {
    'AI', 'LLM', 'LLMs', 'GPT', 'GPT-4', 'GPT-5', 'Claude', 'Gemini', 'OpenAI', 'Anthropic',
    'Google', 'Microsoft', 'Meta', 'Apple', 'Amazon', 'NVIDIA', 'Tesla', 'SpaceX',
    'Transformer', 'BERT', 'ViT', 'ResNet', 'CNN', 'RNN', 'LSTM', 'GAN', 'Diffusion',
    'NLP', 'CV', 'ML', 'DL', 'RL', 'AGI', 'ASI', 'API', 'SDK', 'URL', 'HTTP', 'RSS',
    'arXiv', 'GitHub', 'Python', 'JavaScript', 'Rust', 'C++', 'SQL', 'JSON', 'XML',
    'RLHF', 'DPO', 'RAG', 'CoT', 'ToT', 'GoT',
    'GPU', 'TPU', 'NPU',
}

def translate_text(text):
    """真正的翻译，保留专有名词"""
    if not text or len(text) < 5:
        return text
    try:
        # 临时替换专有名词
        temp_text = text
        replacements = {}
        for word in sorted(NO_TRANSLATE, key=len, reverse=True):
            if word in temp_text:
                placeholder = f"__T{hash(word) % 10000}__"
                replacements[placeholder] = word
                temp_text = temp_text.replace(word, placeholder)
        
        # 翻译
        if len(temp_text) > 1000:
            temp_text = temp_text[:1000]
        translated = translator.translate(temp_text)
        
        # 恢复专有名词
        for placeholder, word in replacements.items():
            translated = translated.replace(placeholder, word)
        
        return translated
    except Exception as e:
        return text

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI论文速递 {date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.7; color: #333; background: #f5f5f5; padding: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 50px; border-radius: 20px; text-align: center; margin-bottom: 40px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .topic {{ margin-bottom: 50px; }}
        .topic h2 {{ color: #1a1a2e; font-size: 1.6em; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 3px solid #667eea; }}
        .paper {{ background: white; padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 3px 15px rgba(0,0,0,0.08); }}
        .paper h3 {{ color: #333; font-size: 1.1em; margin-bottom: 8px; line-height: 1.5; }}
        .paper .title-cn {{ color: #667eea; font-size: 0.95em; display: block; margin-top: 5px; margin-bottom: 12px; }}
        .paper .meta {{ color: #888; font-size: 0.85em; margin-bottom: 12px; }}
        .paper .abstract {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 12px; }}
        .paper .abstract-en {{ color: #555; font-size: 0.9em; margin-bottom: 8px; line-height: 1.6; }}
        .paper .abstract-cn {{ color: #2563eb; font-size: 0.9em; line-height: 1.6; background: #eff6ff; padding: 10px; border-radius: 6px; }}
        .paper .link a {{ color: #667eea; text-decoration: none; font-weight: 500; }}
        .back {{ text-align: center; margin-bottom: 30px; }}
        .back a {{ color: #667eea; text-decoration: none; font-size: 1.1em; }}
        .footer {{ text-align: center; padding: 40px; color: #888; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back"><a href="/">← 返回首页</a></div>
        <div class="header">
            <h1>📚 AI论文速递 {date}</h1>
            <p>7大主题 × 10篇 | 由 <a href="/" style="color:white">伊利虾</a> 🦐 自动整理</p>
        </div>
'''

topic_info = [('LLM','🧠','大语言模型'),('CV','🖼️','计算机视觉'),('多模态','🎨','多模态学习'),('新数据集','📊','新数据集'),('模型压缩','✂️','模型压缩'),('综述','📝','综述论文'),('RL','🎮','强化学习')]

for topic, emoji, topic_cn in topic_info:
    cur = conn.execute("SELECT * FROM papers WHERE topic=? ORDER BY RANDOM() LIMIT 10", (topic,))
    papers = [dict(r) for r in cur.fetchall()]
    html += f'<div class="topic"><h2>{emoji} {topic_cn}</h2>'
    for i, p in enumerate(papers, 1):
        title = p['title'].replace('<', '&lt;').replace('>', '&gt;')
        title_cn = translate_text(title)
        abstract = p['abstract'][:500].replace('<', '&lt;').replace('>', '&gt;')
        abstract_cn = translate_text(abstract)
        
        # 每翻译几篇休息一下
        if i % 3 == 0:
            time.sleep(1)
        html += f'''<div class="paper">
<h3>{i}. {title}<span class="title-cn">➡️ {title_cn}</span></h3>
<p class="meta">👤 {p['authors'][:50]}</p>
<div class="abstract">
<p class="abstract-en">📄 {abstract}...</p>
<p class="abstract-cn">📄 {abstract_cn}...</p>
</div>
<p class="link"><a href="{p["url"]}" target="_blank">🔗 查看论文 →</a></p>
</div>'''
    html += '</div>'

html += f'<div class="footer"><p>数据来源: arXiv | 每天8点自动更新</p></div></body></html>'

path = f'/Users/xuyili/.openclaw/workspace/docs/papers-{date}.html'
with open(path, 'w', encoding='utf-8') as f:
    f.write(html)
conn.close()
print(f"✅ 生成 papers-{date}.html")
PYEOF

# 3. 推送到GitHub
cd /Users/xuyili/.openclaw/workspace/docs
git add papers-$(date +\%Y-\%m-\%d).html
git commit -m "Papers $(date +\%Y-\%m-\%d)"
git push origin main

echo "🎉 完成！"
