#!/usr/bin/env python3
"""RSS抓取脚本 - 从订阅源抓取文章并存入数据库"""

import feedparser
import sqlite3
import time
import os
from datetime import datetime
from rss_feeds import FEEDS

DB_PATH = os.path.expanduser("~/.openclaw/db/rss.db")

def init_db():
    """初始化数据库"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        link TEXT UNIQUE,
        summary TEXT,
        source TEXT,
        author TEXT,
        published_at TEXT,
        fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
        is_ai INTEGER DEFAULT 0
    )''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_published ON articles(published_at)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_is_ai ON articles(is_ai)')
    conn.commit()
    return conn

def is_ai_related(title, summary):
    """判断是否与AI相关"""
    ai_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
        'llm', 'gpt', 'claude', 'gemini', 'chatgpt', 'openai', 'anthropic',
        'transformer', 'neural', 'model', 'training', 'dataset', 'benchmark',
        'nlp', 'nlu', 'nlg', 'gpt-', 'model', 'prompt', 'rlhf', 'dpo',
        'diffusion', 'gpt-4', 'gpt-5', 'Claude', 'Llama', 'Mistral',
        'agent', 'coding', 'programming', 'software', 'developer'
    ]
    text = (title + ' ' + (summary or '')).lower()
    return any(kw in text for kw in ai_keywords)

def parse_date(date_str):
    """解析日期"""
    if not date_str:
        return None
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str).isoformat()
    except:
        return None

def fetch_feed(url):
    """抓取单个RSS源"""
    try:
        feed = feedparser.parse(url)
        return feed.feed.get('title', url), feed.entries
    except Exception as e:
        print(f"  ❌ {url}: {e}")
        return url, []

def main():
    print(f"📰 开始抓取 {len(FEEDS)} 个RSS源...")
    conn = init_db()
    c = conn.cursor()
    
    ai_count = 0
    total_count = 0
    
    for i, url in enumerate(FEEDS):
        print(f"[{i+1}/{len(FEEDS)}] 抓取 {url[:50]}...")
        source, entries = fetch_feed(url)
        
        for entry in entries:
            total_count += 1
            title = entry.get('title', '')
            link = entry.get('link', '')
            
            # 改进：优先获取content:encoded（通常包含完整内容）
            summary = ''
            if hasattr(entry, 'content'):
                for content in entry.content:
                    if hasattr(content, 'value'):
                        summary = content.value
                        break
            
            # 其次尝试 content:encoded
            if not summary:
                summary = entry.get('content_encoded', '')
            
            # 最后用summary/description
            if not summary:
                summary = entry.get('summary', '') or entry.get('description', '')
            
            # 清理HTML标签
            import re
            summary = re.sub(r'<[^>]+>', '', summary)  # 移除HTML标签
            summary = re.sub(r'\s+', ' ', summary).strip()  # 合并空白
            
            author = entry.get('author', '')
            published = parse_date(entry.get('published', ''))
            
            # 检查是否AI相关
            if is_ai_related(title, summary):
                ai_count += 1
                try:
                    # 不再截断，保留完整摘要
                    c.execute('''INSERT OR IGNORE INTO articles 
                        (title, link, summary, source, author, published_at, is_ai)
                        VALUES (?, ?, ?, ?, ?, ?, 1)''',
                        (title, link, summary, source, author, published))
                except:
                    pass
        
        conn.commit()
        time.sleep(0.3)  # 避免请求过快
    
    print(f"\n✅ 完成! 共抓取 {total_count} 篇文章, 其中 {ai_count} 篇可能与AI相关")
    conn.close()

if __name__ == "__main__":
    main()
