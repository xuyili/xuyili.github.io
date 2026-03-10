#!/usr/bin/env python3
"""从数据库生成每日AI新闻网页 - 使用模型翻译"""

import sqlite3
import os
import time
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator

DB_PATH = os.path.expanduser("~/.openclaw/db/rss.db")
OUTPUT_DIR = os.path.expanduser("~/.openclaw/workspace/docs")

# 初始化翻译器
translator = GoogleTranslator(source='en', target='zh-CN')

# 不翻译的专有名词
NO_TRANSLATE = {
    'AI', 'LLM', 'LLMs', 'GPT', 'GPT-4', 'GPT-5', 'Claude', 'Gemini', 'OpenAI', 'Anthropic',
    'Google', 'Microsoft', 'Meta', 'Apple', 'Amazon', 'NVIDIA', 'Tesla', 'SpaceX',
    'Transformer', 'BERT', 'ViT', 'ResNet', 'CNN', 'RNN', 'LSTM', 'GAN', 'Diffusion',
    'NLP', 'CV', 'ML', 'DL', 'RL', 'AGI', 'ASI', 'API', 'SDK', 'URL', 'HTTP', 'RSS',
    'arXiv', 'GitHub', 'Python', 'JavaScript', 'Rust', 'C++', 'SQL', 'JSON', 'XML',
    'Ilya', 'Sutskever', 'Sam', 'Altman', 'Dario', 'Amodei', 'Demis', 'Hassabis',
    'Marcus', 'Gary', 'Dwarkesh', 'Patel', 'Simon', 'Willison', 'Gwern',
    'Pentagon', 'CIA', 'NSA', 'FBI', 'US', 'UK', 'EU', 'China', 'Chinese', 'Chinese',
    'NLP', 'UE', 'SA', 'EU', 'UK', 'US', 'UCLA', 'MIT', 'Stanford', 'Harvard',
    'RLHF', 'DPO', 'RAG', 'CoT', 'ToT', 'GoT', 'IoU', 'mAP', 'F1', 'GPU', 'TPU', 'NPU',
    'Q&A', 'Q&A', 'SaaS', 'PaaS', 'IaaS', 'IT', 'OT', 'CT', 'IoT', 'VR', 'AR', 'MR', 'XR',
    '3D', '2D', '4K', '8K', 'HD', 'FHD', 'QHD', 'UHD', 'SD', 'HD', 'RGB', 'CMYK', 'HSV',
    'CEO', 'CTO', 'COO', 'CFO', 'CMO', 'CIO', 'CPO', 'VP', 'Director', 'Manager', 'Lead',
    'PhD', 'MBA', 'MS', 'BS', 'BA', 'MA', 'JD', 'MD', 'DO', 'RN', 'NP', 'PA', 'RD',
}

def translate(text):
    """翻译文本，保留专有名词"""
    if not text:
        return ""
    try:
        # 临时替换专有名词
        temp_text = text
        replacements = {}
        for word in sorted(NO_TRANSLATE, key=len, reverse=True):  # 先处理长的
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
        print(f"翻译错误: {e}")
        return text

def get_articles(limit=50):
    """从数据库获取最新AI相关文章"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 获取所有AI文章，按发布时间排序取最新的
    c.execute('''SELECT title, link, summary, source, author, published_at 
                FROM articles 
                WHERE is_ai = 1
                ORDER BY published_at DESC
                LIMIT ?''', (limit,))
    
    articles = c.fetchall()
    conn.close()
    return articles

def generate_html(articles):
    """生成HTML页面"""
    today = datetime.now().strftime("%Y年%m月%d日")
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Daily Brief - {today}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.8; color: #333; background: #f5f5f5; padding: 40px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 50px; border-radius: 20px; text-align: center; margin-bottom: 40px; }}
        .header h1 {{ font-size: 3em; margin-bottom: 15px; }}
        .header p {{ font-size: 1.3em; opacity: 0.9; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .article {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .article h2 {{ color: #333; font-size: 1.15em; margin-bottom: 8px; line-height: 1.4; }}
        .article h2 .title-cn {{ color: #667eea; font-weight: 500; font-size: 0.9em; display: block; margin-top: 8px; }}
        .article .meta {{ color: #888; font-size: 0.9em; margin-bottom: 12px; }}
        .article .summary {{ color: #555; margin-bottom: 12px; font-size: 0.95em; line-height: 1.6; }}
        .article .summary-cn {{ color: #2563eb; font-size: 0.9em; line-height: 1.7; background: #eff6ff; padding: 15px; border-radius: 8px; margin-bottom: 12px; }}
        .article .link a {{ color: #667eea; text-decoration: none; font-weight: 500; }}
        .footer {{ text-align: center; padding: 40px; color: #888; }}
        .back {{ text-align: center; margin-bottom: 25px; }}
        .back a {{ color: #667eea; text-decoration: none; font-size: 1.1em; }}
        @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back"><a href="/">← 返回首页</a></div>
        <div class="header">
            <h1>📰 AI Daily Brief</h1>
            <p>{today} · {len(articles)}篇 · 由 <a href="/" style="color:white">伊利虾</a> 🦐 自动整理 · 全文翻译</p>
        </div>
'''
    
    # 分两列显示
    for i, art in enumerate(articles):
        title, link, summary, source, author, published = art
        
        # 翻译标题和摘要
        title_cn = translate(title)
        summary_cn = translate(summary[:1500]) if summary else ''
        
        date = published[:10] if published else ''
        
        if i % 2 == 0:
            if i > 0:
                html += '</div>\n'
            html += '<div class="two-col">\n'
        
        html += f'''        <div class="article">
            <h2>{title}<span class="title-cn">{title_cn}</span></h2>
            <p class="meta">📅 {date} · 👤 {author or 'Unknown'} · 📡 {source}</p>
            <p class="summary">{summary[:250]}...</p>
            <p class="summary-cn">📝 {summary_cn}</p>
            <p class="link"><a href="{link}" target="_blank">🔗 查看原文 →</a></p>
        </div>
'''
        
        # 每翻译几篇休息一下，避免被限流
        if i > 0 and i % 5 == 0:
            time.sleep(1)
    
    if articles:
        html += '</div>\n'
    
    html += '''        <div class="footer">
            <p>🦐 伊利虾 | 由 OpenClaw 驱动 | ''' + today + '''</p>
        </div>
    </div>
</body>
</html>'''
    
    return html

def main():
    print("📰 正在从数据库生成每日AI早报（全文翻译）...")
    
    articles = get_articles(50)
    print(f"   获取到 {len(articles)} 篇AI相关文章")
    
    if not articles:
        print("   ⚠️ 没有找到最近的文章")
        return
    
    print("   🔄 开始翻译文章...")
    html = generate_html(articles)
    
    # 保存文件
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(OUTPUT_DIR, f"ai-{today}.html")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"   ✅ 已生成: {output_path}")
    
    # Git提交
    os.chdir(OUTPUT_DIR)
    os.system('git add .')
    os.system(f'git commit -m "Daily AI brief {today} with translation"')
    os.system('git push origin main')
    print("   ✅ 已推送到GitHub")

if __name__ == "__main__":
    main()
