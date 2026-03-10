#!/bin/bash

# 重试函数
retry() {
    local max_attempts=$1
    local delay=$2
    shift 2
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "[尝试 $attempt/$max_attempts] $@"
        if "$@"; then
            echo "✅ 成功"
            return 0
        fi
        echo "❌ 失败，${delay}秒后重试..."
        sleep $delay
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
    echo "❌ 达到最大重试次数"
    return 1
}

echo "========================================="
echo "开始每日任务 $(date)"
echo "========================================="

# 1. 抓取RSS新闻
echo "📰 抓取AI新闻..."
retry 2 5 python3 /Users/xuyili/.openclaw/workspace/rss-to-db.py || echo "⚠️ RSS抓取失败，跳过"
retry 2 5 python3 /Users/xuyili/.openclaw/workspace/generate-daily.py || echo "⚠️ AI Daily生成失败，跳过"

# 2. 抓取论文
echo "📚 抓取论文..."
retry 2 10 bash /Users/xuyili/.openclaw/workspace/daily-papers.sh

# 3. DeepSeek翻译
echo "🌐 翻译论文和AI新闻..."
DATE=$(date +\%Y-\%m-\%d)
python3 << 'PYEOF'
import re
import json
import openai
import time

client = openai.OpenAI(
    api_key="sk-1e8ec851901a49869d4086aab83e33e6",
    base_url="https://api.deepseek.com"
)

def translate_file(filepath):
    """翻译HTML文件中的摘要"""
    with open(filepath, 'r') as f:
        html = f.read()
    
    # 检查是否已有翻译
    if 'abstract-cn' in html or 'summary-cn' in html:
        print(f"  {filepath} 已有翻译，跳过")
        return
    
    # 提取英文摘要
    pattern = r'(<p class="summary">📄 |class="abstract-en">📄 )(.*?)(</p>)'
    matches = list(re.finditer(pattern, html, re.DOTALL))
    
    if not matches:
        print(f"  {filepath} 无英文摘要")
        return
    
    print(f"  翻译 {len(matches)} 篇...")
    
    translations = []
    for i, match in enumerate(matches):
        en_text = match.group(2)
        if len(en_text) > 3000:
            en_text = en_text[:3000]
        
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"完整翻译为中文：\n{en_text}\n中文："}],
                temperature=0.3
            )
            cn = resp.choices[0].message.content.strip()
            translations.append(cn)
        except Exception as e:
            print(f"  翻译错误: {e}")
            translations.append("翻译失败")
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(matches)}")
        
        time.sleep(0.8)
    
    # 替换/添加中文摘要
    for i, cn in enumerate(translations):
        if cn == "翻译失败":
            continue
        old = f'class="abstract-en">📄 {matches[i].group(2)}</p>'
        new = f'class="abstract-en">📄 {matches[i].group(2)}</p>\n<p class="abstract-cn">📄 {cn}</p>'
        html = html.replace(old, new, 1)
        
        old2 = f'<p class="summary">📄 {matches[i].group(2)}</p>'
        new2 = f'<p class="summary">📄 {matches[i].group(2)}</p>\n<p class="summary-cn">📄 {cn}</p>'
        html = html.replace(old2, new2, 1)
    
    with open(filepath, 'w') as f:
        f.write(html)
    print(f"  {filepath} 翻译完成!")

# 翻译论文
papers_file = f"/Users/xuyili/.openclaw/workspace/docs/papers-{DATE}.html"
try:
    translate_file(papers_file)
except Exception as e:
    print(f"论文翻译错误: {e}")

# 翻译AI新闻
ai_file = f"/Users/xuyili/.openclaw/workspace/docs/ai-{DATE}.html"
try:
    translate_file(ai_file)
except Exception as e:
    print(f"AI新闻翻译错误: {e}")

print("翻译完成!")
PYEOF

# 3. 推送到GitHub
echo "🚀 推送到GitHub..."
cd /Users/xuyili/.openclaw/workspace/docs
retry 3 5 git add ai-$(date +\%Y-\%m-\%d).html papers-$(date +\%Y-\%m-\%d).html archive.html index.html 2>/dev/null || true
retry 3 5 git add papers-$(date +\%Y-\%m-\%d).html archive.html index.html
retry 3 5 git commit -m "Daily update $(date +\%Y-\%m-\%d)" || echo "⚠️ 无新内容可提交"
retry 3 5 git push origin main

echo "🎉 每日任务完成!"
