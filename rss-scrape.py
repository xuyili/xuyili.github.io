#!/usr/bin/env python3
"""
并行抓取 RSS feeds 并筛选最有价值的 50 篇文章
"""
import feedparser
import html
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# RSS feeds 列表
FEEDS = [
    ("simonwillison.net", "https://simonwillison.net/atom/everything/"),
    ("jeffgeerling.com", "https://www.jeffgeerling.com/blog.xml"),
    ("seangoedecke.com", "https://www.seangoedecke.com/rss.xml"),
    ("krebsonsecurity.com", "https://krebsonsecurity.com/feed/"),
    ("daringfireball.net", "https://daringfireball.net/feeds/main"),
    ("ericmigi.com", "https://ericmigi.com/rss.xml"),
    ("antirez.com", "http://antirez.com/rss"),
    ("idiallo.com", "https://idiallo.com/feed.rss"),
    ("maurycyz.com", "https://maurycyz.com/index.xml"),
    ("pluralistic.net", "https://pluralistic.net/feed/"),
    ("shkspr.mobi", "https://shkspr.mobi/blog/feed/"),
    ("lcamtuf.substack.com", "https://lcamtuf.substack.com/feed"),
    ("mitchellh.com", "https://mitchellh.com/feed.xml"),
    ("dynomight.net", "https://dynomight.net/feed.xml"),
    ("utcc.utoronto.ca/~cks", "https://utcc.utoronto.ca/~cks/space/blog/?atom"),
    ("xeiaso.net", "https://xeiaso.net/blog.rss"),
    ("devblogs.microsoft.com/oldnewthing", "https://devblogs.microsoft.com/oldnewthing/feed"),
    ("righto.com", "https://www.righto.com/feeds/posts/default"),
    ("lucumr.pocoo.org", "https://lucumr.pocoo.org/feed.atom"),
    ("skyfall.dev", "https://skyfall.dev/rss.xml"),
    ("garymarcus.substack.com", "https://garymarcus.substack.com/feed"),
    ("rachelbythebay.com", "https://rachelbythebay.com/w/atom.xml"),
    ("overreacted.io", "https://overreacted.io/rss.xml"),
    ("timsh.org", "https://timsh.org/rss/"),
    ("johndcook.com", "https://www.johndcook.com/blog/feed/"),
    ("gilesthomas.com", "https://gilesthomas.com/feed/rss.xml"),
    ("matklad.github.io", "https://matklad.github.io/feed.xml"),
    ("derekthompson.org", "https://www.theatlantic.com/feed/author/derek-thompson/"),
    ("evanhahn.com", "https://evanhahn.com/feed.xml"),
    ("terriblesoftware.org", "https://terriblesoftware.org/feed/"),
    ("rakhim.exotext.com", "https://rakhim.exotext.com/rss.xml"),
    ("joanwestenberg.com", "https://joanwestenberg.com/rss"),
    ("xania.org", "https://xania.org/feed"),
    ("micahflee.com", "https://micahflee.com/feed/"),
    ("nesbitt.io", "https://nesbitt.io/feed.xml"),
    ("construction-physics.com", "https://www.construction-physics.com/feed"),
    ("tedium.co", "https://feed.tedium.co/"),
    ("susam.net", "https://susam.net/feed.xml"),
    ("entropicthoughts.com", "https://entropicthoughts.com/feed.xml"),
    ("buttondown.com/hillelwayne", "https://buttondown.com/hillelwayne/rss"),
    ("dwarkesh.com", "https://www.dwarkeshpatel.com/feed"),
    ("borretti.me", "https://borretti.me/feed.xml"),
    ("wheresyoured.at", "https://www.wheresyoured.at/rss/"),
    ("jayd.ml", "https://jayd.ml/feed.xml"),
    ("minimaxir.com", "https://minimaxir.com/index.xml"),
    ("geohot.github.io", "https://geohot.github.io/blog/feed.xml"),
    ("paulgraham.com", "http://www.aaronsw.com/2002/feeds/pgessays.rss"),
    ("filfre.net", "https://www.filfre.net/feed/"),
    ("blog.jim-nielsen.com", "https://blog.jim-nielsen.com/feed.xml"),
    ("dfarq.homeip.net", "https://dfarq.homeip.net/feed/"),
    ("jyn.dev", "https://jyn.dev/atom.xml"),
    ("geoffreylitt.com", "https://www.geoffreylitt.com/feed.xml"),
    ("downtowndougbrown.com", "https://www.downtowndougbrown.com/feed/"),
    ("brutecat.com", "https://brutecat.com/rss.xml"),
    ("eli.thegreenplace.net", "https://eli.thegreenplace.net/feeds/all.atom.xml"),
    ("abortretry.fail", "https://www.abortretry.fail/feed"),
    ("fabiensanglard.net", "https://fabiensanglard.net/rss.xml"),
    ("oldvcr.blogspot.com", "https://oldvcr.blogspot.com/feeds/posts/default"),
    ("bogdanthegeek.github.io", "https://bogdanthegeek.github.io/blog/index.xml"),
    ("hugotunius.se", "https://hugotunius.se/feed.xml"),
    ("gwern.net", "https://gwern.substack.com/feed"),
    ("berthub.eu", "https://berthub.eu/articles/index.xml"),
    ("chadnauseam.com", "https://chadnauseam.com/rss.xml"),
    ("simone.org", "https://simone.org/feed/"),
    ("it-notes.dragas.net", "https://it-notes.dragas.net/feed/"),
    ("beej.us", "https://beej.us/blog/rss.xml"),
    ("hey.paris", "https://hey.paris/index.xml"),
    ("danielwirtz.com", "https://danielwirtz.com/rss.xml"),
    ("matduggan.com", "https://matduggan.com/rss/"),
    ("refactoringenglish.com", "https://refactoringenglish.com/index.xml"),
    ("worksonmymachine.substack.com", "https://worksonmymachine.substack.com/feed"),
    ("philiplaine.com", "https://philiplaine.com/index.xml"),
    ("steveblank.com", "https://steveblank.com/feed/"),
    ("bernsteinbear.com", "https://bernsteinbear.com/feed.xml"),
    ("danieldelaney.net", "https://danieldelaney.net/feed"),
    ("troyhunt.com", "https://www.troyhunt.com/rss/"),
    ("herman.bearblog.dev", "https://herman.bearblog.dev/feed/"),
    ("tomrenner.com", "https://tomrenner.com/index.xml"),
    ("blog.pixelmelt.dev", "https://blog.pixelmelt.dev/rss/"),
    ("martinalderson.com", "https://martinalderson.com/feed.xml"),
    ("danielchasehooper.com", "https://danielchasehooper.com/feed.xml"),
    ("chiark.greenend.org.uk/~sgtatham", "https://www.chiark.greenend.org.uk/~sgtatham/quasiblog/feed.xml"),
    ("grantslatton.com", "https://grantslatton.com/rss.xml"),
    ("experimental-history.com", "https://www.experimental-history.com/feed"),
    ("anildash.com", "https://anildash.com/feed.xml"),
    ("aresluna.org", "https://aresluna.org/main.rss"),
    ("michael.stapelberg.ch", "https://michael.stapelberg.ch/feed.xml"),
    ("miguelgrinberg.com", "https://blog.miguelgrinberg.com/feed"),
    ("keygen.sh", "https://keygen.sh/blog/feed.xml"),
    ("mjg59.dreamwidth.org", "https://mjg59.dreamwidth.org/data/rss"),
    ("computer.rip", "https://computer.rip/rss.xml"),
    ("tedunangst.com", "https://www.tedunangst.com/flak/rss"),
]

def clean_html(text):
    """清理 HTML 标签"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:500]

def is_recent(entry, days=7):
    """检查文章是否在最近几天内发布"""
    try:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            from time import mktime
            pub_dt = datetime.fromtimestamp(mktime(entry.published_parsed))
            return (datetime.now() - pub_dt).days <= days
    except:
        pass
    return True

# 关键词权重
BOOST_KEYWORDS = [
    'AI', 'LLM', 'GPT', 'Claude', 'OpenAI', 'model', 'training', 'inference',
    'agent', 'workflow', 'automation', 'tool', 'function calling',
    'security', 'vulnerability', 'exploit', 'attack',
    'performance', 'optimization', 'benchmark', 'eval',
    'architecture', 'design', 'system', 'infrastructure',
    'learning', 'neural', 'transformer', 'attention',
    'programming', 'developer', 'code', 'debug',
    'hardware', 'GPU', 'TPU',
    'startup', 'business', 'product', 'market',
    'ethics', 'policy', 'regulation', 'privacy',
]

def calculate_score(entry, source_name):
    """计算文章权重分数"""
    score = 0
    
    high_quality_sources = [
        'Simon Willison', 'Paul Graham', 'Gwern', 'Bruce Schneier',
        'Overreacted', 'The Old New Thing', 'Dwarkesh Patel',
        'Gary Marcus', 'Mitchell Hashimoto', 'Antirez', 'Krebs',
    ]
    for qs in high_quality_sources:
        if qs.lower() in source_name.lower():
            score += 3
    
    title = clean_html(entry.get('title', '')).lower()
    for kw in BOOST_KEYWORDS:
        if kw.lower() in title:
            score += 2
    
    content = clean_html(entry.get('summary', entry.get('description', ''))).lower()
    for kw in BOOST_KEYWORDS:
        if kw.lower() in content:
            score += 0.5
    
    if is_recent(entry, days=2):
        score += 2
    elif is_recent(entry, days=7):
        score += 1
    
    if len(content) > 300:
        score += 1
    
    return score

def fetch_feed(name, url):
    """抓取单个 feed"""
    try:
        feed = feedparser.parse(url)
        entries = []
        for entry in feed.entries[:10]:
            # 为每个 entry 手动添加来源信息
            entry_with_source = {
                'title': entry.get('title', ''),
                'summary': entry.get('summary', entry.get('description', '')),
                'link': entry.get('link', ''),
                'published_parsed': getattr(entry, 'published_parsed', None),
                'feed_title': name,
            }
            entries.append(entry_with_source)
        return entries
    except Exception as e:
        return []

def main():
    print("🔄 正在并行抓取 RSS feeds...")
    
    all_entries = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_feed, name, url): (name, url) 
                   for name, url in FEEDS}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                entries = future.result()
                all_entries.extend(entries)
                print(f"  ✓ {i}/{len(FEEDS)}")
            except Exception as e:
                pass
    
    print(f"📰 共获取 {len(all_entries)} 篇文章")
    
    # 按权重排序
    scored_entries = [(e, calculate_score(e, e.get('feed_title', ''))) for e in all_entries]
    scored_entries.sort(key=lambda x: x[1], reverse=True)
    
    # 取前50篇
    top_50 = [e for e, s in scored_entries[:50]]
    
    print(f"\n✅ 精选 50 篇文章:")
    for i, entry in enumerate(top_50, 1):
        title = clean_html(entry.get('title', ''))
        source = entry.get('feed_title', 'Unknown')
        print(f"{i}. [{source}] {title}")
    
    # 保存到 JSON
    import json
    result = []
    for entry in top_50:
        result.append({
            'title': clean_html(entry.get('title', '')),
            'summary': clean_html(entry.get('summary', ''))[:300],
            'link': entry.get('link', ''),
            'source': entry.get('feed_title', 'Unknown'),
        })
    
    with open('/Users/xuyili/.openclaw/workspace/docs/rss-top50.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n💾 已保存到 rss-top50.json")

if __name__ == "__main__":
    main()