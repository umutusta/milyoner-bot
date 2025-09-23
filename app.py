import os, re, json, requests, concurrent.futures
from flask import Flask, request, jsonify

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERPAPI_KEY    = os.environ.get("f59608f6288ba71a5fc6567b0ef4de425c4bef3cb0eed5acf28e318b4b0e2a44", "")

SESS = requests.Session()
SESS.headers.update({"User-Agent": "milyoner-bot/1.0"})

# -------------------- OpenAI helper --------------------

def openai_chat(messages, model="gpt-4o-mini", temperature=0, max_tokens=220):
    r = SESS.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": temperature,
              "messages": messages, "max_tokens": max_tokens},
        timeout=18
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# -------------------- Utils --------------------

def is_negated(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [" değil", " değildir", " olmayan",
                                 " hangisi değildir", " hariç",
                                 " dışındadır", " yanlış olan"])

# -------------------- Retrieval --------------------

def wiki_search(question: str, limit: int = 3):
    url = "https://tr.wikipedia.org/w/api.php"
    res = SESS.get(url, params={
        "action": "query", "list": "search", "srsearch": question,
        "format": "json", "utf8": 1, "srlimit": limit, "srprop": "snippet"
    }, timeout=6)
    items = (res.json().get("query", {}).get("search", []) if res.ok else [])
    extracts = []
    for it in items[:limit]:
        title = it["title"]
        sum_url = f"https://tr.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        s = SESS.get(sum_url, timeout=6)
        if s.ok:
            j = s.json()
            extracts.append({"title": title, "extract": j.get("extract","")})
    return extracts

def serpapi_google(query: str, num: int = 8):
    if not SERPAPI_KEY:
        return []
    r = SESS.get("https://serpapi.com/search.json", params={
        "engine": "google", "q": query, "hl": "tr", "gl": "tr",
        "num": num, "api_key": SERPAPI_KEY
    }, timeout=8)
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

def serpapi_ai_overview(question: str):
    """Fetch Google AI Overview (if available)."""
    if not SERPAPI_KEY:
        return []
    try:
        r = SESS.get("https://serpapi.com/search.json", params={
            "engine": "google", "q": question,
            "hl": "en", "gl": "us", "num": "8",
            "api_key": SERPAPI_KEY
        }, timeout=8)
        r.raise_for_status()
        j = r.json()
        aio = j.get("ai_overview")
        if not aio:
            return []
        # If token exists, fetch dedicated AIO
        if "page_token" in aio:
            r2 = SESS.get("https://serpapi.com/search.json", params={
                "engine": "google_ai_overview",
                "page_token": aio["page_token"],
                "api_key": SERPAPI_KEY
            }, timeout=8)
            r2.raise_for_status()
            aio = r2.json().get("ai_overview", aio)

        snippets = []
        for tb in aio.get("text_blocks", []):
            if tb.get("text"):
                snippets.append(tb["text"])
        for ref in aio.get("references", []):
            title, link = ref.get("title",""), ref.get("link","")
            if title or link:
                snippets.append(f"{title} {link}".strip())
        return snippets
    except Exception as e:
        return [f"AIO fetch failed: {e}"]

# -------------------- Scoring --------------------

def simple_score(option_text: str, blobs: list[str]) -> int:
    opt = option_text.lower()
    toks = [t for t in re.split(r"[^0-9a-zçğıöşüâîû]+", opt) if len(t) > 2]
    score = 0
    for b in blobs:
        b = b.lower()
        for t in toks:
            if t in b:
                score += 1
    return score

def build_evidence(question: str, options):
    # Run wiki + google in parallel
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            "wiki": ex.submit(wiki_search, question),
        }
        for label, text in options.items():
            futs[f"g_{label}"] = ex.submit(serpapi_google, f"{question} {text}")
        for k, f in futs.items():
            results[k] = f.result()

    # Collect blobs
    global_blobs = []
    for w in results.get("wiki", []):
        global_blobs.append(f"{w['title']}: {w['extract']}")
    blobs_per_opt = {k: [] for k in options}
    for k, items in results.items():
        if k.startswith("g_"):
            label = k.split("_",1)[1]
            for it in items:
                txt = f"{it['title']} — {it['snippet']}"
                blobs_per_opt[label].append(txt)
                global_blobs.append(txt)

    # Scores
    scores = {}
    for label, text in options.items():
        scores[label] = simple_score(text, blobs_per_opt[label] + global_blobs)

    return scores, global_blobs

# -------------------- Flask endpoints --------------------

@app.get("/health")
def health():
    return jsonify(ok=True, has_openai=bool(OPENAI_API_KEY), has_serp=bool(SERPAPI_KEY))

@app.post("/answer")
def answer():
    try:
        data = request.get_json(silent=True) or {}
        raw = (data.get("ocr_text") or "").strip()
        if not raw:
            return jsonify(error="missing ocr_text"), 400

        # Step A: cleanup OCR → normalized MCQ
        sys_cleanup = (
            "Convert raw OCR into a clean Turkish MCQ with A–D options.\n"
            "Output exactly:\n"
            "Question: ...?\nA) ...\nB) ...\nC) ...\nD) ..."
        )
        cleaned = openai_chat(
            [{"role":"system","content":sys_cleanup},
             {"role":"user","content":raw}],
            model="gpt-4o-mini", max_tokens=320
        )

        # Parse question & options
        qm = re.search(r"Question:\s*(.+?\?)", cleaned, flags=re.S|re.I)
        question = (qm.group(1).strip() if qm else cleaned.strip())
        def opt(label):
            m = re.search(rf"^{label}\)\s*(.+)$", cleaned, flags=re.M)
            return (m.group(1).strip() if m else "")
        options = { "A": opt("A"), "B": opt("B"),
                    "C": opt("C"), "D": opt("D") }

        # Step B: first retrieval & scoring
        scores, corpus = build_evidence(question, options)
        neg = is_negated(question)

        # Step C: GPT final answer (first pass)
        sys_final = (
            "You are a strict verifier. Choose the correct option USING ONLY the supplied evidence.\n"
            "Rules:\n"
            "1) If the question is negated (değil/olmayan/hariç/dışında/yanlış), pick the option with the WEAKEST support.\n"
            "2) Otherwise pick the option with the STRONGEST support.\n"
            "3) Break ties using numeric scores.\n"
            "4) Output one line: 'A) Option text [CONF:##]'."
        )
        user_final = {
            "question": question,
            "options": options,
            "negated": neg,
            "scores": scores,
            "snippets": corpus[:20]
        }
        first = openai_chat(
            [{"role":"system","content":sys_final},
             {"role":"user","content":json.dumps(user_final, ensure_ascii=False)}],
            model="gpt-4.1", max_tokens=80
        )

        ans_match  = re.search(r"^[ABCD]\)\s.*?(?=\s*\[CONF:)", first, flags=re.S|re.M)
        conf_match = re.search(r"\[CONF:\s*(\d{1,3})\s*\]", first)
        answer_line = (ans_match.group(0).strip() if ans_match else first.strip())
        conf = int(conf_match.group(1)) if conf_match else None

        # Step D: If confidence is low, try AI Overview
        if conf is None or conf < 70:
            aio_snips = serpapi_ai_overview(question)
            if aio_snips:
                user_final["snippets"] = (corpus + aio_snips)[:30]
                retry = openai_chat(
                    [{"role":"system","content":sys_final},
                     {"role":"user","content":json.dumps(user_final, ensure_ascii=False)}],
                    model="gpt-4.1", max_tokens=80
                )
                ans_match  = re.search(r"^[ABCD]\)\s.*?(?=\s*\[CONF:)", retry, flags=re.S|re.M)
                conf_match = re.search(r"\[CONF:\s*(\d{1,3})\s*\]", retry)
                answer_line = (ans_match.group(0).strip() if ans_match else retry.strip())
                conf = int(conf_match.group(1)) if conf_match else conf

        return jsonify(answer=answer_line,
                       confidence=conf,
                       cleaned_mcq=f"Question: {question}\nA) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}")

    except Exception as e:
        return jsonify(error=type(e).__name__, message=str(e)), 500
