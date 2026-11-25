"""Builds a larger text corpus per domain by asking the running llama-server (Inkling 2 Mini) to write it.

Usage: py scripts/gen_corpus.py --domain chat --target 12000 [--port 8080]
Appends the model's replies (thinking + answer) to data/corpus/<domain>.txt until about <target> tokens are reached.
Using the model's own output is deliberate: it is exactly the text the model routes during real use.
"""
import argparse, json, os, random, time, urllib.request

PROMPTS = {
    "chat": [
        "My friend forgot my birthday. How should I bring it up without starting a fight?",
        "I want to start running but I get bored fast. Any ideas?",
        "What should I cook tonight with eggs, rice, onions and some leftover chicken?",
        "How do I politely say no to extra work from my manager?",
        "Plan a cheap but fun weekend in a city I have never visited.",
        "I can't sleep before exams. What helps?",
        "Explain to me like a friend why people enjoy horror movies.",
        "How do I make new friends after moving to a new city?",
        "Give me honest advice: should I buy a used car or a new one?",
        "My plants keep dying. What am I probably doing wrong?",
        "How can I stop scrolling my phone so much in the evening?",
        "What are good questions to ask on a first date?",
        "I have a job interview tomorrow for a junior developer role. How should I prepare tonight?",
        "How do I keep a conversation going when I am shy?",
        "Suggest a weekly routine to keep a small apartment clean without much effort.",
        "What is a good way to apologise to someone I hurt a long time ago?",
    ],
    "math": [
        "Solve step by step: a shop gives 20% off, then another 15% off the new price. What is the total discount?",
        "Find all real solutions of x^2 - 7x + 10 = 0 and check them.",
        "A train travels 300 km at 75 km/h and returns at 100 km/h. What is the average speed for the whole trip?",
        "Prove that the sum of two odd numbers is always even.",
        "What is the probability of getting at least one six in four rolls of a fair die? Show the work.",
        "Compute the derivative of f(x) = x^3 * sin(x) and explain each step.",
        "How many ways can 5 people sit in a row if two of them refuse to sit next to each other?",
        "Integrate x * e^x dx step by step.",
        "A rectangle has perimeter 40 and area 96. Find its sides.",
        "Explain why the square root of 2 is irrational with a full proof.",
        "Solve the system: 2x + 3y = 13 and 5x - y = 7.",
        "What is the sum of the first 50 terms of the arithmetic sequence 3, 7, 11, ...?",
        "Find the area of a triangle with sides 7, 8 and 9 using Heron's formula.",
        "If log2(x) + log2(x - 2) = 3, find x.",
        "A bank pays 5% interest compounded yearly. How long until money doubles? Show the calculation.",
        "How many diagonals does a polygon with 12 sides have? Derive the formula.",
    ],
    "arabic": [
        "اشرح لي بطريقة بسيطة كيف يعمل الإنترنت.",
        "ما هي أفضل طريقة لتنظيم الوقت أثناء الدراسة الجامعية؟",
        "اكتب قصة قصيرة عن طفل يجد قطة ضائعة في يوم ممطر.",
        "ما الفرق بين الطاقة الشمسية وطاقة الرياح؟",
        "كيف أبدأ مشروعاً صغيراً بميزانية محدودة؟",
        "اشرح لماذا تتغير الفصول الأربعة خلال السنة.",
        "أعطني نصائح لتعلم لغة جديدة بسرعة.",
        "ما هي فوائد القراءة اليومية؟",
        "كيف أكتب سيرة ذاتية جيدة لأول وظيفة؟",
        "اشرح مفهوم الذكاء الاصطناعي لشخص لا يعرف شيئاً عن التقنية.",
        "ما هي أهم العادات الصحية التي يجب أن أبدأ بها؟",
        "صف مدينة قديمة كأنك تتجول فيها.",
        "كيف أتعامل مع ضغط العمل والتوتر؟",
        "ما هي أسباب التلوث وكيف يمكن الحد منه؟",
        "اشرح كيف تعمل الحاسبة داخل الهاتف بلغة بسيطة.",
        "اكتب رسالة شكر لمعلم أثّر في حياتي.",
    ],
}


def ask(port, prompt, max_tokens, seed):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.8, "seed": seed}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n" + (m.get("content") or ""), d["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=PROMPTS)
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    os.makedirs("data/corpus", exist_ok=True)
    out = f"data/corpus/{a.domain}.txt"
    total, i, t0 = 0, 0, time.time()
    prompts = PROMPTS[a.domain][:]
    random.Random(1).shuffle(prompts)
    with open(out, "w", encoding="utf-8") as f:
        while total < a.target:
            p = prompts[i % len(prompts)]
            text, n = ask(a.port, p, a.max_tokens, seed=i)
            f.write(f"User: {p}\nAssistant: {text.strip()}\n\n")
            f.flush()
            total += n
            i += 1
            print(f"{a.domain}: {i} replies, ~{total} tokens, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
