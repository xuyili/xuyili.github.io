import requests
import json
import time

def translate(text):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {'client': 'gtx', 'sl': 'en', 'tl': 'zh-CN', 'dt': 't', 'q': text[:500]}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            result = r.json()
            return ''.join([x[0] for x in result[0] if x[0]])
    except:
        pass
    return text

# 读取论文
try:
    with open('/Users/xuyili/.openclaw/workspace/docs/papers-2026-03-07.html') as f:
        content = f.read()
    
    # 简单替换 title-cn 的内容
    # 这里简化处理，实际可以用更好的方式
    
    print("翻译功能就绪")
except Exception as e:
    print(f"Error: {e}")
