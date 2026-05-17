from typing import List, Dict, Optional
import json
import logging
import os
import re
import time
from urllib.parse import urljoin, urlparse
from collections import Counter, defaultdict

import requests
from bs4 import BeautifulSoup
# optional playwright fallback for dynamic pages
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from shl_agent.ingestion.cleaning import normalize_text, extract_tags

logger = logging.getLogger(__name__)


class CatalogScraper:
    """Scraper for SHL "Individual Test Solutions" catalog entries.

    The scraper is intentionally conservative: it uses requests + BeautifulSoup
    and exposes a simple interface for scraping and saving cleaned JSON.

    Usage:
        scraper = CatalogScraper()
        records = scraper.scrape()
        scraper.save(records, "data/processed/catalog.json")
    """

    def __init__(
        self,
        base_url: str = "https://www.shl.com/",
        category_keyword: str = "Individual Test",
        session: Optional[requests.Session] = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url
        self.category_keyword = category_keyword
        self.timeout = timeout
        self.session = session or self._init_session()

    def _init_session(self) -> requests.Session:
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "shl-agent-scraper/0.1 (+https://example.com)"})
        return s

    def _fetch(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _find_candidate_links(self, html: str) -> List[str]:
        """Find links likely to be Individual Test Solutions pages.

        This uses flexible heuristics: link text and URL path matching.
        """
        soup = BeautifulSoup(html, "html.parser")
        links = []
        # primary: look for product/assessment tiles or article links
        selectors = ["a[href*='/products/assessments']", "a[href*='/assessments']", "a[href*='/products/']", "article a[href]"]
        for sel in selectors:
            for a in soup.select(sel):
                href = a.get("href")
                if not href:
                    continue
                href = urljoin(self.base_url, href)
                text = (a.get_text(" ") or "").strip()
                # include when link text or url path suggests an assessment
                if re.search(r"individual|assessment|test|sjt|opq|gsa", text, re.I) or re.search(r"/assessments|/products/assessments|/products/", href, re.I):
                    links.append(href)
        # fallback: any link with 'assessment' or 'test' in path
        if not links:
            for a in soup.find_all("a", href=True):
                href = urljoin(self.base_url, a["href"])
                if re.search(r"assessment|assessments|test|sjt|opq|gsa", href, re.I):
                    links.append(href)
        # also parse JSON-LD for product/ItemList entries
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                j = json.loads(s.string or "{}")
            except Exception:
                continue
            # product or itemList
            if isinstance(j, dict):
                if j.get("@type") in ("Product", "ProductModel", "CreativeWork"):
                    url = j.get("url") or j.get("@id")
                    if url:
                        links.append(urljoin(self.base_url, url))
                if j.get("@type") == "ItemList":
                    for it in j.get("itemListElement", []) or []:
                        u = it.get("url") or (it.get("item", {}) or {}).get("url")
                        if u:
                            links.append(urljoin(self.base_url, u))
            elif isinstance(j, list):
                for el in j:
                    if not isinstance(el, dict):
                        continue
                    url = el.get("url") or el.get("@id")
                    if url:
                        links.append(urljoin(self.base_url, url))
        # dedupe while preserving order
        seen = set()
        out = []
        for l in links:
            cl = self._canonicalize_url(l)
            if cl not in seen:
                seen.add(cl)
                out.append(cl)
        return out

    def _discover_sitemaps_from_robots(self) -> List[str]:
        robots_url = urljoin(self.base_url, "robots.txt")
        try:
            txt = self._fetch(robots_url)
            if not txt:
                return []
            sitemaps = re.findall(r"Sitemap:\s*(\S+)", txt, re.I)
            return [self._canonicalize_url(s) for s in sitemaps]
        except Exception:
            return []

    def _canonicalize_url(self, url: str) -> str:
        # remove query, fragment, trailing slash
        p = urlparse(url)
        path = p.path.rstrip('/')
        return f"{p.scheme}://{p.netloc}{path}"

    def _parse_assessment(self, html: str, url: str) -> Dict:
        """Parse an assessment page into a structured record.

        This function uses robust fallbacks and returns normalized fields.
        """
        soup = BeautifulSoup(html, "html.parser")
        # title
        h = soup.find(["h1"]) or soup.find(["h2"]) or soup.find(class_=re.compile(r"heading|title", re.I))
        title = h.get_text(strip=True) if h else ""

        # description: first long paragraph under title
        description = ""
        if soup.find("meta", attrs={"name": "description"}):
            description = soup.find("meta", attrs={"name": "description"})["content"].strip()
        else:
            p = soup.find("p")
            if p:
                description = p.get_text(" ", strip=True)

        # gather textual area for feature extraction
        content_text = " ".join([t.get_text(" ", strip=True) for t in soup.find_all(["p", "li", "dd"])])

        # naive extraction for fields using regex heuristics
        duration = self._extract_field(content_text, r"(\d+\s?(?:minutes|mins|hours))")
        assessment_type = self._extract_field(content_text, r"(personality|ability|aptitude|numerical|verbal|inductive)", default="Unknown")
        remote_support = bool(re.search(r"remote|online|proctored|remote proctored|invigilat", content_text, re.I))

        # skills measured: extract comma-separated segments or bullets near keywords
        skills = self._extract_skills(soup, content_text)

        # target roles: try to find sections like "suitable for" or "target"
        target_roles = self._extract_field(content_text, r"(suitable for|ideal for|target (?:roles|audience):?)[\s:-]+([^\n\.]+)", group=2)

        # try to detect assessment family / breadcrumb
        family = ""
        bc = soup.find("nav", class_=re.compile(r"breadcrumb|breadcrumbs", re.I)) or soup.find(class_=re.compile(r"breadcrumb|breadcrumbs", re.I))
        if bc:
            family = " ".join([t.get_text(strip=True) for t in bc.find_all("a")])

        # detect job levels (junior/mid/senior/lead)
        jl = None
        m = re.search(r"(junior|mid|senior|lead|entry-level|manager)", content_text, re.I)
        if m:
            jl = m.group(1).lower()

        record = {
            "name": normalize_text(title) or normalize_text(self._fallback_name_from_url(url)),
            "url": self._canonicalize_url(url),
            "description": description or content_text[:400],
            "assessment_type": assessment_type.title() if isinstance(assessment_type, str) else "Unknown",
            "duration": duration or "",
            "remote": remote_support,
            "skills": skills,
            "target_roles": target_roles or "",
            "assessment_family": normalize_text(family) if family else "",
            "job_level": jl or "",
        }

        # heuristics for specific assessment pages
        is_specific = False
        path = urlparse(url).path.strip('/')
        depth = len(path.split('/'))
        has_specific_slug = bool(re.search(r"-[a-z0-9]+", path))
        has_duration = bool(duration)
        has_skills = bool(skills)
        has_title_marker = bool(re.search(r"assessment|questionnaire|simulation|test|questionnaire|survey", title, re.I))
        # require stronger signals to mark a page as a specific assessment
        # prefer explicit duration or combination of slug+skills, or deeper path
        if (depth >= 4) or has_duration or (has_specific_slug and has_skills):
            is_specific = True

        record["is_specific_assessment"] = bool(is_specific)

        # normalized tags for retrieval
        record["tags"] = extract_tags(record["name"] + " " + record["description"] + " " + " ".join(record["skills"]))

        return record

    def _fallback_name_from_url(self, url: str) -> str:
        path = urlparse(url).path.strip("/")
        return path.split("/")[-1].replace("-", " ")

    def _extract_field(self, text: str, pattern: str, group: int = 1, default: Optional[str] = "") -> Optional[str]:
        m = re.search(pattern, text, re.I)
        if not m:
            return default
        return (m.group(group) if group and m.groups() else m.group(0)).strip()

    def _extract_skills(self, soup: BeautifulSoup, fallback_text: str) -> List[str]:
        # try to find a heading that contains 'skills' and extract the following list
        for header in soup.find_all(["h2", "h3", "strong"]):
            if header and re.search(r"skill", header.get_text(), re.I):
                # find sibling ul/li
                ul = header.find_next("ul")
                if ul:
                    return [li.get_text(strip=True) for li in ul.find_all("li")]
        # fallback: use simple keyword extraction on fallback_text
        tags = extract_tags(fallback_text)
        return tags[:10]

    def scrape(self, landing_path: str = "", max_pages: int = 5) -> List[Dict]:
        """Main scraping entrypoint.

        - Fetches `base_url + landing_path`
        - Finds candidate assessment links using heuristics
        - Fetches each candidate and parses the structured record
        - Deduplicates by canonical URL
        """
        landing_url = urljoin(self.base_url, landing_path)
        logger.debug("Fetching landing page %s", landing_url)
        landing_html = self._fetch(landing_url)
        # if landing fetch empty and playwright available, try JS-rendered fetch
        if not landing_html and PLAYWRIGHT_AVAILABLE:
            logger.debug("Landing page empty; trying Playwright for %s", landing_url)
            landing_html = self._fetch_with_playwright(landing_url)
        if not landing_html:
            logger.error("Could not fetch landing page: %s", landing_url)
            return []
        # discover sitemaps from robots or common locations
        sitemaps = self._discover_sitemaps_from_robots()
        for sitemap_path in ["sitemap.xml", "sitemap_index.xml"]:
            s = urljoin(self.base_url, sitemap_path)
            if s not in sitemaps:
                sitemaps.append(s)

        candidate_links = []
        # try parsing sitemaps first
        for surl in sitemaps:
            try:
                stext = self._fetch(surl)
                if not stext:
                    continue
                locs = re.findall(r"<loc>(.*?)</loc>", stext, re.I)
                for l in locs:
                    if re.search(r"/products/assessments|/assessments/", l, re.I):
                        candidate_links.append(self._canonicalize_url(l))
                if candidate_links:
                    logger.debug("Found %d sitemap candidate links from %s", len(candidate_links), surl)
                    break
            except Exception:
                continue

        # fallback to in-page discovery
        if not candidate_links:
            candidate_links = self._find_candidate_links(landing_html)
        logger.debug("Found %d initial candidate links", len(candidate_links))

        # attempt to follow pagination: look for pagination links on landing page
        soup = BeautifulSoup(landing_html or "", "html.parser")
        # follow pagination aggressively and site search endpoints
        pag_links = [urljoin(self.base_url, a["href"]) for a in soup.find_all("a", href=True) if re.search(r"page|pagination|next", a.get_text(" ") or "", re.I)]
        # attempt site search endpoints
        search_candidates = []
        for q in ["assessments", "assessment", "sjt", "cognitive", "personality"]:
            for sp in ["/search/?q=", "/search?q=", "/?s=", "/search/?query="]:
                search_candidates.append(urljoin(self.base_url, f"{sp}{q}"))
        follow_pages = (pag_links + search_candidates)[: max(0, max_pages * 2)]
        for pl in follow_pages:
            ph = self._fetch(pl)
            if not ph and PLAYWRIGHT_AVAILABLE:
                ph = self._fetch_with_playwright(pl)
            if ph:
                more = self._find_candidate_links(ph)
                for m in more:
                    if m not in candidate_links:
                        candidate_links.append(m)

        # recursive discovery: BFS crawl candidate pages and collect deeper assessment links
        discovered = list(candidate_links)
        visited = set(discovered)
        idx = 0
        # increase crawl breadth for aggressive discovery
        max_total = max_pages * 200
        while idx < len(discovered) and len(discovered) < max_total:
            cur = discovered[idx]
            idx += 1
            html = self._fetch(cur)
            if not html and PLAYWRIGHT_AVAILABLE:
                html = self._fetch_with_playwright(cur)
            if not html:
                continue
            try:
                more = self._find_candidate_links(html)
            except Exception:
                more = []
            for m in more:
                if m in visited:
                    continue
                # avoid adding top-level category/index pages
                p = urlparse(m).path.rstrip('/')
                parts = [seg for seg in p.split('/') if seg]
                # allow some category pages to be explored but prefer deeper pages
                if len(parts) <= 2 and re.search(r"assessments?$", p, re.I):
                    # likely a category/index page; deprioritize but still visit
                    # add only if we haven't exceeded a small ratio
                    if len(discovered) > max_pages * 5:
                        continue
                visited.add(m)
                discovered.append(m)

        candidate_links = discovered
        logger.debug("After recursive discovery, total candidate links: %d", len(candidate_links))

        records: List[Dict] = []
        seen_urls = set()
        total_crawled = 0
        rejected_category = 0
        duplicates = 0
        depth_dist = Counter()
        meta_counts = Counter()
        # limit to unique candidate links up to max_pages*20
        for url in candidate_links[: max(0, max_pages * 50)]:
            if url in seen_urls:
                continue
            total_crawled += 1
            logger.debug("Fetching assessment page %s", url)
            html = self._fetch(url)
            if not html and PLAYWRIGHT_AVAILABLE:
                html = self._fetch_with_playwright(url)
            if not html:
                logger.warning("Skipping %s due to fetch error", url)
                continue
            try:
                rec = self._parse_assessment(html, url)
            except Exception as exc:
                logger.exception("Failed to parse %s: %s", url, exc)
                continue
            # canonicalize url
            canonical = rec.get("url")
            if not canonical:
                continue
            if canonical in seen_urls:
                duplicates += 1
                continue
            # skip category/index pages that are not specific assessments
            if not rec.get('is_specific_assessment'):
                logger.debug("Skipping non-specific/category page %s", canonical)
                rejected_category += 1
                # still record depth distribution
                path = urlparse(canonical).path.strip('/')
                depth_dist[len([p for p in path.split('/') if p])] += 1
                continue
            # ensure this appears to be an Individual Test Solution by checking keywords
            if not re.search(r"individual|assessment|test|sjt|opq|gsa|assessment", rec.get("name", "") + " " + rec.get("description", ""), re.I):
                logger.debug("Skipping non-ITS page %s", canonical)
                continue
            seen_urls.add(canonical)
            records.append(rec)
            # metadata completeness
            if rec.get('duration'):
                meta_counts['duration'] += 1
            if rec.get('skills'):
                meta_counts['skills'] += 1
            if rec.get('assessment_type') and rec.get('assessment_type') != 'Unknown':
                meta_counts['assessment_type'] += 1
            if rec.get('job_level'):
                meta_counts['job_level'] += 1
            if rec.get('remote'):
                meta_counts['remote'] += 1
            # depth
            path = urlparse(canonical).path.strip('/')
            depth_dist[len([p for p in path.split('/') if p])] += 1
            # rate-limit defensively
            time.sleep(0.2)

        # produce analysis
        analysis = {
            'total_urls_crawled': total_crawled,
            'accepted_assessments': len(records),
            'rejected_category_pages': rejected_category,
            'duplicate_count': duplicates,
            'metadata_counts': dict(meta_counts),
            'depth_distribution': dict(depth_dist),
        }
        try:
            os.makedirs('data/processed', exist_ok=True)
            with open('data/processed/catalog_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)
        except Exception:
            logger.exception('Failed to write catalog_analysis.json')

        return records

    def save(self, records: List[Dict], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # prevent duplicates by url
        out = []
        seen = set()
        for r in records:
            url = r.get("url")
            if url in seen:
                continue
            seen.add(url)
            out.append(r)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    def _fetch_with_playwright(self, url: str, timeout: int = 15) -> Optional[str]:
        if not PLAYWRIGHT_AVAILABLE:
            return None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=timeout * 1000)
                content = page.content()
                browser.close()
                return content
        except Exception as exc:
            logger.warning("Playwright fetch failed for %s: %s", url, exc)
            return None

    def load_local(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

