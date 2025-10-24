import os, re, json, time, math, logging, requests, concurrent.futures
from functools import lru_cache
from flask import Flask, request, jsonify

app = Flask(__name__)

# ----------- LOGGER SETUP -----------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bilmatik-bot")

# ----------- ENV / CLIENTS -----------
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
SERPAPI_KEY         = os.getenv("SERPAPI_KEY", "")
PERPLEXITY_API_KEY  = os.getenv("PERPLEXITY_API_KEY", "")

SESS = requests.Session()
SESS.headers.update({"User-Agent": "bilmatik-bot/1.3"})

TOTAL_BUDGET_SEC = 9.5

# ----------- UTILITIES -----------

NEG_PATTERNS = [
    " değil", " değildir", " olmayan", " hangisi değildir", " hariç",
    " dışındadır", " yanlış olan", " değildir?"
]

def now(): return time.perf_counter()
def time_left(start): return max(0.0, TOTAL_BUDGET_SEC - (now() - start))
def clamp(n, lo, hi): return max(lo, min(hi, n))

def is_negated(q: str) -> bool:
    ql = (" " + q.lower() + " ")
    return any(p in ql for p in NEG_PATTERNS)

def only_letter(s: str) -> str:
    m = re.search(r"\b([ABCD])\b", s.strip(), flags=re.I)
    return m.group(1).upper() if m else s.strip()

# ----------- OPENAI -----------

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
    except Exception as e:
        log.warning(f"Wiki search failed: {e}")
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
        return [
            {"title": it.get("title",""), "snippet": it.get("snippet","")}
            for it in j.get("organic_results", [])
        ]
    except Exception as e:
        log.warning(f"SerpAPI failed: {e}")
        return []

def perplexity_mcq(question: str, options: dict, timeout: float = 4.0):
    if not PERPLEXITY_API_KEY or timeout <= 0.1:
        return None
    try:
        prompt = (
            "Türkçe bilgi yarışması sorusu. "
            "Web araması yaparak doğru şıkkı bul ve sadece harf döndür (A/B/C/D).\n\n"
            f"Soru: {question}\n"
            f"A) {options.get('A','')}\n"
            f"B) {options.get('B','')}\n"
            f"C) {options.get('C','')}\n"
            f"D) {options.get('D','')}\nCevap:"
        )
        r = SESS.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": "sonar", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 4, "temperature": 0, "return_citations": True},
            timeout=timeout
        )
        if not r.ok:
            return None
        txt = r.json()["choices"][0]["message"]["content"]
        return only_letter(txt)
    except Exception as e:
        log.warning(f"Perplexity error: {e}")
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
    results = {}
    start_t = now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {"wiki": ex.submit(wiki_search_cached, question)}
        if SERPAPI_KEY:
            for label, text in options.items():
                futs[f"g_{label}"] = ex.submit(serpapi_google, f"{question} {text}")
        for k, f in futs.items():
            try:
                results[k] = f.result(timeout=3.0)
            except Exception as e:
                log.warning(f"{k} retrieval failed: {e}")
                results[k] = []
    elapsed = now() - start_t
    log.info(f"Retrieval done in {elapsed:.2f}s")

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
    scores = {label: simple_score(text, blobs_per_opt[label] + global_blobs)
              for label, text in options.items()}
    return scores, global_blobs[:40]

# ----------- FLASK ENDPOINT -----------

@app.get("/health")
def health():
    return jsonify(ok=True,
                   has_openai=bool(OPENAI_API_KEY),
                   has_serp=bool(SERPAPI_KEY),
                   has_perplexity=bool(PERPLEXITY_API_KEY))

@app.post("/answer")
def answer():
    start = now()
    try:
        data = request.get_json(silent=True) or {}
        raw = (data.get("ocr_text") or "").strip()
        if not raw:
            return jsonify(error="missing ocr_text"), 400
        log.info(f"Received OCR text ({len(raw)} chars)")

        # STEP A – cleanup
        sys_cleanup = (
            "OCR'den gelen karışık metni düzgün Türkçe MCQ formatına çevir:\n"
            "Question: ...?\nA) ...\nB) ...\nC) ...\nD) ..."
        )
        t0 = now()
        cleaned = openai_chat(
            [{"role":"system","content":sys_cleanup},
             {"role":"user","content":raw}],
            model="gpt-4o-mini",
            max_tokens=220,
            timeout=min(3.5, time_left(start))
        )
        log.info(f"StepA: Cleaned OCR in {now()-t0:.2f}s")

        # Parse
        qm = re.search(r"Question:\s*(.+?\?)", cleaned, flags=re.S|re.I)
        question = (qm.group(1).strip() if qm else raw[:220].strip())
        def opt(label):
            m = re.search(rf"^{label}\)\s*(.+)$", cleaned, flags=re.M)
            return (m.group(1).strip() if m else "")
        options = { "A": opt("A"), "B": opt("B"), "C": opt("C"), "D": opt("D") }

        # STEP B – retrieval
        scores, corpus = build_evidence(question, options, time_left(start))
        log.info(f"StepB: Scores: {scores}")

        neg = is_negated(question)
        log.info(f"Negation detected: {neg}")

        # STEP C – decision
        sys_final = (
            "Verilen kanıta göre doğru şıkkı seç.\n"
            "Eğer olumsuz (değil/olmayan) varsa en düşük destekli şıkkı seç.\n"
            "Sadece A/B/C/D döndür."
        )
        user_payload = {
            "question": question, "options": options,
            "negated": neg, "scores": scores, "snippets": corpus[:20]
        }
        t1 = now()
        first = openai_chat(
            [{"role":"system","content":sys_final},
             {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)}],
            model="gpt-4o-mini",
            max_tokens=4,
            timeout=min(2.2, time_left(start))
        )
        answer_letter = only_letter(first)
        log.info(f"StepC: Model picked {answer_letter} in {now()-t1:.2f}s")

        # Confidence
        top, bot = max(scores.values()), min(scores.values())
        conf = 50 + clamp(7 * abs(top - bot), 0, 45)
        if neg: conf = max(40, conf - 5)

        # STEP D – fallback
        if conf < 70 and time_left(start) > 4.2:
            log.info("Low confidence -> using Perplexity fallback")
            px = perplexity_mcq(question, options, timeout=min(4.0, time_left(start)-0.1))
            if px in ("A","B","C","D"):
                answer_letter, conf = px, max(conf, 78)
                log.info(f"StepD: Perplexity changed answer -> {px}")

        cleaned_mcq = f"Question: {question}\nA) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}"
        log.info(f"Final: {answer_letter} ({int(conf)}%) | total {now()-start:.2f}s")

        return jsonify(answer=answer_letter, confidence=int(conf), cleaned_mcq=cleaned_mcq)

    except Exception as e:
        log.exception("Error in /answer")
        return jsonify(error=type(e).__name__, message=str(e)), 500
