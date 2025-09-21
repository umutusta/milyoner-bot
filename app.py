import os, json, re, requests
from flask import Flask, request, jsonify

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")

app = Flask(__name__)

def openai_chat(messages, model="gpt-4.1", temperature=0, max_tokens=64):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": temperature,
              "messages": messages, "max_tokens": max_tokens}
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

@app.post("/answer")
def answer():
    data = request.get_json(force=True)
    ocr = (data or {}).get("ocr_text","").strip()
    if not ocr:
        return jsonify(error="missing ocr_text"), 400

    sys_cleanup = (
        "Your task is to transform raw OCR text into a clean multiple-choice question in Turkish.\n"
        "- Detect the question sentence (ending with '?').\n"
        "- Collect A,B,C,D options and reorder as A,B,C,D.\n"
        "- Output format:\n\nQuestion: ...\nA) ...\nB) ...\nC) ...\nD) ..."
    )
    cleaned = openai_chat([
        {"role":"system","content": sys_cleanup},
        {"role":"user","content": ocr}
    ], max_tokens=300)

    m = re.search(r"Question:\s*(.+?\?)", cleaned, flags=re.S|re.I)
    short_q = (m.group(1) if m else cleaned)[:140]

    serp_json = {}
    if SERPAPI_KEY:
        try:
            sr = requests.get(
                "https://serpapi.com/search.json",
                params={"engine":"google","q":short_q,"hl":"tr","gl":"tr","num":"8","api_key":SERPAPI_KEY},
                timeout=8
            )
            serp_json = sr.json()
        except Exception:
            serp_json = {}

    sys_ans = (
        "You are a quiz assistant. You will be given a Turkish MCQ (A–D) and a brief Google result JSON.\n"
        "- Output only the correct answer in the format: \"A) Option text\".\n"
        "- No explanations.\n"
        "- Append confidence like [CONF:0-100]."
    )
    final_out = openai_chat([
        {"role":"system","content": sys_ans},
        {"role":"user","content": f"MCQ:\n{cleaned}\n\nSEARCH JSON:\n{json.dumps(serp_json)[:8000]}"}
    ], max_tokens=64)

    ans_match = re.search(r"^[ABCD]\)\s.*?(?=\s*\[CONF:)", final_out, flags=re.S|re.M)
    conf_match = re.search(r"\[CONF:\s*(\d{1,3})\s*\]", final_out)
    answer_line = ans_match.group(0).strip() if ans_match else final_out.strip()
    conf = int(conf_match.group(1)) if conf_match else None

    return jsonify(answer=answer_line, confidence=conf, cleaned_mcq=cleaned)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
