import os, re, json, time, math, requests, concurrent.futures
from functools import lru_cache
from flask import Flask, request, jsonify

app = Flask(__name__)

# ----------- ENV / CLIENTS -----------
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
SERPAPI_KEY         = os.getenv("SERPAPI_KEY", "")           # <- fixed
PERPLEXITY_API_KEY  = os.getenv("PERPLEXITY_API_KEY", "")   # optional fallback

SESS = requests.Session()
SESS.headers.update({"User-Agent": "bilmatik-bot/1.2"})

# Global time budget per request (Bilmatik shows 10s; keep some headroom)
TOTAL_BUDGET_SEC = 9.5

# ----------- UTILITIES -----------

NEG_PATTERNS = [
    " değil", " değildir", " olmayan", " hangisi değildir", " hariç",
    " dışındadır", " yanlış olan", " değildir?"
]

def is_negated(q: str) -> bool:
    ql = (" " + q.lower() + " ")
    return any(p in ql for p in NEG_PATTERNS)

def now() -> float:
    return time.perf_counter()

def time_left(start: float) -> float:
    return max(0.0, TOTAL_BUDGET_SEC - (now() - start))

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

def only_letter(s: str) -> str:
    m = re.search(r"\b([ABCD])\b", s.strip(), flags=re.I)
    return m.group(1).upper() if m else s.strip()

# ----------- OPENAI (tight & deterministic) -----------

def openai_chat(messages, model="gpt-4o-mini", temperature=0, max_tokens=48, timeout=4.5):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    r = SESS.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": temperature,
              "messages": messages, "max_tokens": max_tokens},
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ----------- RETRIEVAL -----------

@lru_cache(maxsize=512)
def wiki_search_cached(question: str, limit: int = 3):
    try:
        url = "https://tr.wikipedia.org/w/api.php"
        res = SESS.get(url, params={
            "action": "query", "list": "search", "srsearch": question,
            "format": "json", "utf8": 1, "srlimit": limit, "srprop": "snippet"
        }, timeout=2.3)
        items = (res.json().get("query", {}).get("search", []) if res.ok else [])
        extracts = []
        for it in items[:limit]:
            title = it["title"]
            sum_url = f"https://tr.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
            s = SESS.get(sum_url, timeout=2.0)
            if s.ok:
                j = s.json()
                extracts.append({"title": title, "extract": j.get("extract","")})
        return extracts
    except Exception:
        return []

def serpapi_google(query: str, num: int = 6):
    if not SERPAPI_KEY:
        return []
    try:
        r = SESS.get("https://serpapi.com/search.json", params={
            "engine": "google", "q": query, "hl": "tr", "gl": "tr",
            "num": num, "api_key": SERPAPI_KEY
        }, timeout=2.8)
        if not r.ok:
            return []
        j = r.json()
        out = []
        for it in j.get("organic_results", []):
            out.append({
                "title": it.get("title",""),
                "snippet": it.get("snippet",""),
                "link": it.get("link","")
            })
        return out
    except Exception:
        return []

def perplexity_mcq(question: str, options: dict, timeout: float = 4.0):
    """
    Ultra-fast web-grounded fallback. Returns a single letter if successful.
    Requires PERPLEXITY_API_KEY.
    """
    if not PERPLEXITY_API_KEY or timeout <= 0.1:
        return None

    try:
        prompt = (
            "Soru ve şıklar çoktan seçmeli bir bilgi yarışmasından. "
            "Web araması yaparak hızlıca doğru cevabı bul ve SADECE harf döndür (A/B/C/D). "
            "Açıklama yazma.\n\n"
            f"Soru: {question}\n"
            f"A) {options.get('A','')}\n"
            f"B) {options.get('B','')}\n"
            f"C) {options.get('C','')}\n"
            f"D) {options.get('D','')}\n"
            "Cevap:"
        )
        r = SESS.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",                 # fast, search-enabled
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4,
                "temperature": 0,
                "return_citations": True
            },
            timeout=timeout
        )
        if not r.ok:
            return None
        txt = r.json()["choices"][0]["message"]["content"]
        return only_letter(txt)
    except Exception:
        return None

# ----------- SCORING -----------

def simple_score(option_text: str, blobs: list[str]) -> int:
    opt = option_text.lower()
    toks = [t for t in re.split(r"[^0-9a-zçğıöşüâîû]+", opt) if len(t) > 2]
    score = 0
    for b in blobs:
        bl = b.lower()
        for t in toks:
            if t in bl:
                score += 1
    return score

def build_evidence(question: str, options, time_budget_left: float):
    # Cap time for retrieval to avoid overruns
    # Wiki (2.3s) + parallel Google per option (2.8s each) under a thread pool
    results = {}
    max_workers = 1 + sum(1 for _ in options)  # wiki + 4 options
    budget_ok = time_budget_left > 2.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=(max_workers if budget_ok else 2)) as ex:
        futs = {"wiki": ex.submit(wiki_search_cached, question)}
        if budget_ok and SERPAPI_KEY:
            for label, text in options.items():
                futs[f"g_{label}"] = ex.submit(serpapi_google, f"{question} {text}")
        for k, f in futs.items():
            try:
                results[k] = f.result(timeout=3.0)
            except Exception:
                results[k] = []

    # Collect blobs
    global_blobs = []
    for w in results.get("wiki", []) or []:
        if w.get("extract"):
            global_blobs.append(f"{w['title']}: {w['extract']}")
    blobs_per_opt = {k: [] for k in options}
    for k, items in (results.items() if results else []):
        if k.startswith("g_"):
            label = k.split("_",1)[1]
            for it in items or []:
                txt = f"{it.get('title','')} — {it.get('snippet','')}"
                blobs_per_opt[label].append(txt)
                global_blobs.append(txt)

    # Scores
    scores = {}
    for label, text in options.items():
        scores[label] = simple_score(text, (blobs_per_opt[label] + global_blobs)[:40])

    return scores, global_blobs[:40]

# ----------- FLASK -----------

@app.get("/health")
def health():
    return jsonify(
        ok=True,
        has_openai=bool(OPENAI_API_KEY),
        has_serp=bool(SERPAPI_KEY),
        has_perplexity=bool(PERPLEXITY_API_KEY)
    )

@app.post("/answer")
def answer():
    start = now()
    try:
        data = request.get_json(silent=True) or {}
        raw = (data.get("ocr_text") or "").strip()
        if not raw:
            return jsonify(error="missing ocr_text"), 400

        # A) Normalize OCR → MCQ (tight format, fast model)
        sys_cleanup = (
            "Aşağıdaki ham OCR metnini düzgün bir Türkçe çoktan seçmeli formata çevir.\n"
            "Çıktı ŞABLONU (aynı satırlarla):\n"
            "Question: ...?\nA) ...\nB) ...\nC) ...\nD) ..."
        )
        cleaned = openai_chat(
            [{"role":"system","content":sys_cleanup},
             {"role":"user","content":raw}],
            model="gpt-4o-mini",
            max_tokens=220,
            timeout=min(3.5, time_left(start))
        )

        # Parse
        qm = re.search(r"Question:\s*(.+?\?)", cleaned, flags=re.S|re.I)
        question = (qm.group(1).strip() if qm else raw[:220].strip())
        def opt(label):
            m = re.search(rf"^{label}\)\s*(.+)$", cleaned, flags=re.M)
            return (m.group(1).strip() if m else "")
        options = { "A": opt("A"), "B": opt("B"), "C": opt("C"), "D": opt("D") }

        # If options look empty (rare OCR failure), try to extract letters inline
        if sum(1 for v in options.values() if v) < 2:
            # last-chance: split lines starting with A)/B)/C)/D)
            lines = [ln.strip() for ln in raw.splitlines()]
            for L in ("A)","B)","C)","D)"):
                for ln in lines:
                    if ln.startswith(L):
                        options[L[0]] = ln[len(L):].strip()

        # B) Retrieval & scoring (respect time budget)
        scores, corpus = build_evidence(question, options, time_left(start))
        neg = is_negated(question)

        # C) First pass answer (STRICT: only letter)
        sys_final = (
            "SANA VERİLEN KANIT DIŞINA ÇIKMA.\n"
            "Kurallar:\n"
            "1) Soru olumsuzsa (değil/olmayan/hariç/dışında/yanlış), EN DÜŞÜK desteği olan şıkkı seç.\n"
            "2) Aksi halde EN YÜKSEK desteği olan şıkkı seç.\n"
            "3) Eşitlikte sayısal skorları kullan.\n"
            "4) SADECE harfi döndür: A veya B veya C veya D. Açıklama yazma."
        )
        user_payload = {
            "question": question,
            "options": options,
            "negated": neg,
            "scores": scores,
            "snippets": corpus[:20]
        }
        first = openai_chat(
            [{"role":"system","content":sys_final},
             {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)}],
            model="gpt-4o-mini",
            max_tokens=4,
            timeout=min(2.2, time_left(start))
        )
        answer_letter = only_letter(first)

        # D) Heuristic confidence (fast)
        try:
            top = max(scores.values()) if scores else 0
            bot = min(scores.values()) if scores else 0
            spread = abs(top - bot)
            conf = 50 + clamp(7 * spread, 0, 45)  # 50–95 crude band
            if neg:
                conf = max(40, conf - 5)
        except Exception:
            conf = 60

        # E) Low-confidence web fallback (Perplexity), only if time permits
        if conf < 70 and time_left(start) > 4.2:
            px = perplexity_mcq(question, options, timeout=min(4.0, time_left(start) - 0.1))
            if px in ("A","B","C","D"):
                answer_letter = px
                conf = max(conf, 78)  # bump if we got a clean, grounded letter

        # Final packaging for your Shortcut
        cleaned_mcq = f"Question: {question}\nA) {options.get('A','')}\nB) {options.get('B','')}\nC) {options.get('C','')}\nD) {options.get('D','')}"
        return jsonify(answer=answer_letter, confidence=int(conf), cleaned_mcq=cleaned_mcq)

    except Exception as e:
        return jsonify(error=type(e).__name__, message=str(e)), 500
