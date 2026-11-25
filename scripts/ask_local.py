"""Send one chat message to the local llama-server and keep the conversation history between calls.

Usage: py scripts/ask_local.py "your message"     (add --reset to start a new conversation)
History is stored in build-tmp/phone-chat.json. Prints the model's thinking (short) and its answer.
"""
import json, os, sys, urllib.request

HIST = os.path.join(os.path.dirname(__file__), "..", "build-tmp", "phone-chat.json")


def main():
    args = [a for a in sys.argv[1:] if a != "--reset"]
    if "--reset" in sys.argv or not os.path.exists(HIST):
        msgs = []
    else:
        msgs = json.load(open(HIST, encoding="utf-8"))
    if not args:
        json.dump([], open(HIST, "w", encoding="utf-8"))
        print("(new conversation)")
        return
    msgs.append({"role": "user", "content": " ".join(args)})
    body = {"messages": msgs, "max_tokens": 1500}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    m = d["choices"][0]["message"]
    answer = (m.get("content") or "").strip()
    msgs.append({"role": "assistant", "content": answer})
    json.dump(msgs, open(HIST, "w", encoding="utf-8"), ensure_ascii=False)
    t = d.get("timings", {})
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"[{t.get('predicted_n', 0)} tokens, {t.get('predicted_per_second', 0):.1f} tok/s]")
    print("THINKING:", (m.get("reasoning_content") or "").strip()[:300].replace("\n", " "))
    print("ANSWER:", answer)


if __name__ == "__main__":
    main()
