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

# 3. 推送到GitHub
echo "🚀 推送到GitHub..."
cd /Users/xuyili/.openclaw/workspace/docs
retry 3 5 git add ai-$(date +\%Y-\%m-\%d).html papers-$(date +\%Y-\%m-\%d).html archive.html index.html 2>/dev/null || true
retry 3 5 git add papers-$(date +\%Y-\%m-\%d).html archive.html index.html
retry 3 5 git commit -m "Daily update $(date +\%Y-\%m-\%d)" || echo "⚠️ 无新内容可提交"
retry 3 5 git push origin main

echo "🎉 每日任务完成!"
