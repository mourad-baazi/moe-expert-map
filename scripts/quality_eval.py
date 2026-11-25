"""Measures how closely a model's greedy output follows a reference model's, on prompts not used for pruning.

Step 1 (reference): py scripts/quality_eval.py --save results/quality/full.json   (server running the full model)
Step 2 (candidate): py scripts/quality_eval.py --save results/quality/x.json --ref results/quality/full.json

Metrics vs the reference, per prompt, over 64 greedy tokens:
  match_prefix = tokens identical before the first difference (64 = identical continuation)
  same_8       = first 8 tokens identical (yes/no)
"""
import argparse, json, os, urllib.request

PROMPTS = [
    "The French Revolution began in the year", "Photosynthesis is the process by which",
    "In Python, a list comprehension is", "The main difference between TCP and UDP is",
    "To make a cup of green tea, you should", "The Pythagorean theorem states that",
    "def fibonacci(n):\n    ", "SELECT name, age FROM users WHERE",
    "The human heart has four chambers:", "Mount Everest is located in",
    "A good way to save money every month is", "The derivative of sin(x) is",
    "Once upon a time, in a small village by the sea,", "The Great Wall of China was built to",
    "Machine learning is a field of study that", "The boiling point of water depends on",
    "In 1969, Apollo 11", "To reverse a string in JavaScript, you can",
    "The capital of Spain is Madrid, and the capital of Portugal is", "Electric cars are becoming popular because",
    "A prime number is a number that", "The Amazon rainforest is important because",
    "git commit -m", "The speed of light in a vacuum is approximately",
    "Dear hiring manager, I am writing to apply for", "The mitochondria is often called",
    "for i in range(10):\n    if i % 2 == 0:\n        ", "Shakespeare wrote many plays, including",
    "The three states of matter are", "A healthy breakfast could include",
    "World War II ended in", "Binary search works by",
    "The Eiffel Tower is made of", "Interest rates affect the economy because",
    "The planet Mars is called the Red Planet because", "In chess, the queen can move",
    "To install a Python package, run", "Volcanoes erupt when",
    "The longest bone in the human body is", "Recycling helps the environment by",
]


def complete(port, prompt, n):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "return_tokens": True, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d.get("tokens", []), d.get("content", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.save), exist_ok=True)
    out = [{"prompt": p, **dict(zip(("tokens", "text"), complete(a.port, p, a.n)))} for p in PROMPTS]
    json.dump(out, open(a.save, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.ref:
        ref = json.load(open(a.ref, encoding="utf-8"))
        prefixes, same8 = [], 0
        for r, c in zip(ref, out):
            k = 0
            while k < min(len(r["tokens"]), len(c["tokens"])) and r["tokens"][k] == c["tokens"][k]:
                k += 1
            prefixes.append(k)
            same8 += k >= 8
        print(f"match_prefix mean {sum(prefixes) / len(prefixes):.1f} / {a.n} tokens | identical 64: "
              f"{sum(p >= a.n for p in prefixes)}/{len(prefixes)} | first 8 identical: {same8}/{len(prefixes)}")
    else:
        print("saved reference", a.save)


if __name__ == "__main__":
    main()
