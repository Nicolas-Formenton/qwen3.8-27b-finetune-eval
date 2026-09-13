import time, json
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="eval")

# warmup
c.chat.completions.create(model="m", messages=[{"role":"user","content":"hi"}], max_tokens=8)

prompt = "Explain in detail how GPU memory bandwidth affects LLM inference speed."
t0 = time.time(); n = 0; ttft = None
s = c.chat.completions.create(model="m", messages=[{"role":"user","content":prompt}],
                              max_tokens=256, temperature=0.2, stream=True)
for ch in s:
    d = ch.choices[0].delta.content
    if d:
        if ttft is None: ttft = time.time()-t0
        n += 1
dt = time.time()-t0
print(json.dumps({"tokens": n, "total_s": round(dt,2), "ttft_s": round(ttft or 0,3),
                  "decode_tok_s": round(n/dt,1)}))
