
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext

# ── Configuration & Constants (from parser.py) ────────────────────────────────

MAX_TEXT_CHARS = 250000
MAX_LINKS = 500
MAX_JSONLD = 50

ECOMMERCE_HINTS = {
    "shopify": [r"cdn\.shopify\.com", r"shopify", r"myshopify"],
    "woocommerce": [r"woocommerce", r"wp-content", r"wp-json"],
    "bigcommerce": [r"cdn\d+\.bigcommerce\.com", r"bigcommerce"],
    "magento": [r"mage\/", r"magento"],
    "squarespace": [r"squarespace", r"static1\.squarespace"],
}

# ── Extraction Logic (Ported from parser.py) ──────────────────────────────────

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def same_domain(url_a: str, url_b: str) -> bool:
    a = urlparse(url_a).netloc.lower().replace("www.", "")
    b = urlparse(url_b).netloc.lower().replace("www.", "")
    return a == b

def extract_meta_description(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if meta and meta.get("content"):
        return clean_text(meta["content"])
    og = soup.find("meta", attrs={"property": re.compile("^og:description$", re.I)})
    if og and og.get("content"):
        return clean_text(og["content"])
    return ""

def remove_noise(soup: BeautifulSoup) -> BeautifulSoup:
    soup = BeautifulSoup(str(soup), "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "template"]):
        tag.decompose()
    for el in soup.select("[aria-hidden='true'], .sr-only, .visually-hidden, .hidden, [hidden]"):
        el.decompose()
    return soup

def extract_visible_text_from_soup(soup: BeautifulSoup) -> str:
    cleaned = remove_noise(soup)
    text = cleaned.get_text(" ", strip=True)
    return clean_text(text)[:MAX_TEXT_CHARS]

def extract_nav_text(soup: BeautifulSoup) -> List[str]:
    items: List[str] = []
    candidates = soup.find_all(["nav", "header"])
    if not candidates:
        candidates = [soup]
    seen: Set[str] = set()
    for block in candidates:
        for a in block.find_all(["a", "button"]):
            txt = clean_text(a.get_text(" ", strip=True))
            if not txt or len(txt) > 80:
                continue
            key = txt.lower()
            if key not in seen:
                seen.add(key)
                items.append(txt)
    return items[:200]

def extract_footer_text(soup: BeautifulSoup) -> str:
    footer = soup.find("footer")
    if not footer:
        return ""
    return clean_text(footer.get_text(" ", strip=True))[:40000]

def extract_links(base_url: str, soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
    internal: List[str] = []
    external: List[str] = []
    seen_i: Set[str] = set()
    seen_e: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https"):
            continue
        if same_domain(base_url, full):
            if full not in seen_i:
                seen_i.add(full)
                internal.append(full)
        else:
            if full not in seen_e:
                seen_e.add(full)
                external.append(full)
        if len(internal) >= MAX_LINKS and len(external) >= MAX_LINKS:
            break
    return internal[:MAX_LINKS], external[:MAX_LINKS]

def extract_jsonld(soup: BeautifulSoup) -> List[Any]:
    results: List[Any] = []
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        raw = script.string or script.get_text(strip=True) or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            results.append(json.loads(raw))
        except Exception:
            results.append({"unparsed": raw[:5000]})
        if len(results) >= MAX_JSONLD:
            break
    return results

def extract_script_srcs(base_url: str, soup: BeautifulSoup) -> List[str]:
    srcs: List[str] = []
    seen: Set[str] = set()
    for script in soup.find_all("script", src=True):
        src = urljoin(base_url, script["src"].strip())
        if src not in seen:
            seen.add(src)
            srcs.append(src)
    return srcs[:500]

def detect_tech_hints(html: str, script_srcs: List[str]) -> List[str]:
    corpus = " ".join([html[:60000], " ".join(script_srcs)]).lower()
    hits = []
    for tech, patterns in ECOMMERCE_HINTS.items():
        if any(re.search(p, corpus, re.I) for p in patterns):
            hits.append(tech)
    return hits

# ── Crawlee Implementation ────────────────────────────────────────────────────

async def run_parser(url: str, output_filename: str = "crawlee_result.json"):
    results = []

    crawler = PlaywrightCrawler(
        max_requests_per_crawl=1,
        browser_type='chromium',
        browser_launch_options={
            "args": ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        }
    )

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f"Processing {context.request.url}...")
        
        # Wait for network idle to ensure JS content is loaded
        try:
            await context.page.wait_for_load_state('networkidle', timeout=10000)
        except Exception:
            context.log.warning("Network idle timeout, continuing anyway")
        
        # Scroll to bottom to trigger lazy loading
        await context.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1) 
        
        html = await context.page.content()
        soup = BeautifulSoup(html, "lxml")
        
        # Extraction
        title = await context.page.title()
        meta_description = extract_meta_description(soup)
        visible_text = extract_visible_text_from_soup(soup)
        nav_text = extract_nav_text(soup)
        footer_text = extract_footer_text(soup)
        internal_links, external_links = extract_links(context.request.url, soup)
        jsonld = extract_jsonld(soup)
        script_srcs = extract_script_srcs(context.request.url, soup)
        tech_hints = detect_tech_hints(html, script_srcs)
        
        res = {
            "url": context.request.url,
            "final_url": context.page.url,
            "title": title,
            "meta_description": meta_description,
            "visible_text": visible_text,
            "nav_text": nav_text,
            "footer_text": footer_text,
            "internal_links": internal_links,
            "external_links": external_links,
            "jsonld": jsonld,
            "script_srcs": script_srcs,
            "tech_hints": tech_hints,
            "status": "ok"
        }
        results.append(res)
        await context.push_data(res)

    print(f"Starting crawl for {url}")
    await crawler.run([url])
    
    if results:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results[0], f, indent=2, ensure_ascii=False)
        
        print(f"\n--- EXTRACTION COMPLETE ---")
        print(f"Results saved to: {output_filename}")
        return results[0]
    return None

async def main():
    if len(sys.argv) < 2:
        print("Usage: python crawlee_parser.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    await run_parser(url)

if __name__ == "__main__":
    asyncio.run(main())
