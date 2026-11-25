"""Send a sequence of chat requests to a running llama-server and report which ones come back broken.

Usage: py scripts/server_probe.py plain,plain,tools,plain [--port 8080]
  plain = normal chat message
  tools = same, but with a tool definition attached (like the llama-server web page sends)
  stream = plain message with streaming on
A reply counts as OK if it has any answer or thinking text.
"""
import argparse, json, urllib.request

TOOL = [{"type": "function", "function": {"name": "search_chats", "description": "Search previous chats",
         "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
QUESTIONS = ["Explain in 4 sentences how a refrigerator keeps food cold.",
             "Write a Python function that checks if a number is prime, with a short explanation.",
             "Explain in 4 sentences why the sky is blue.",
             "Give 3 tips for learning a new language, one sentence each."]


def ask(port, i, kind):
    body = {"messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}], "max_tokens": 400,
            "temperature": 0}
    if kind == "tools":
        body["tools"] = TOOL
    if kind == "stream":
        body["stream"] = True
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode("utf-8", "replace")
    if kind == "stream":
        text = ""
        for line in raw.splitlines():
            if line.startswith("data: {"):
                d = json.loads(line[6:])["choices"][0].get("delta", {})
                text += (d.get("content") or "") + (d.get("reasoning_content") or "")
        return bool(text.strip()), text[:80]
    d = json.loads(raw)
    m = d["choices"][0]["message"]
    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
    if m.get("tool_calls"):
        text += " [tool call]"
    t = d.get("timings", {})
    stats = f"{t.get('predicted_per_second', 0):.1f} tok/s"
    if t.get("draft_n"):
        stats += f", drafts accepted {t.get('draft_n_accepted', 0)}/{t['draft_n']}"
    return bool(text.strip()), f"[{stats}] " + text[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    for i, kind in enumerate(a.sequence.split(",")):
        ok, preview = ask(a.port, i, kind)
        print(f"  {i + 1}. {kind:6s} {'OK  ' if ok else 'BROKEN'} {preview!r}")


if __name__ == "__main__":
    main()
