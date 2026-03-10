#!/bin/bash
# Git自动推送脚本 - 带重试机制

RETRY=3
DELAY=5

cd ~/.openclaw/workspace/docs

for i in $(seq 1 $RETRY); do
    echo "[尝试 $i/$RETRY] 正在推送..."
    if git push origin main 2>&1; then
        echo "✅ 推送成功!"
        exit 0
    fi
    echo "❌ 推送失败，${DELAY}秒后重试..."
    sleep $DELAY
done

echo "❌ 推送多次失败，请手动检查"
exit 1
