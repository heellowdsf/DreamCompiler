# Pure text tool (zero ML): corpus -> char indices -> sliding-window CSVs.
base = ("the quick brown fox jumps over the lazy dog. "
        "dream compiles tensors into light. "
        "gradients flow backward through the graph. "
        "attention learns what matters most. ")
corpus = base * 60
chars = sorted(set(corpus)); c2i = {c: i for i, c in enumerate(chars)}
idx = [c2i[c] for c in corpus]; S = 16
rows = [(idx[i:i+S], idx[i+S]) for i in range(len(idx) - S)]
rows = rows[:(len(rows)//32)*32]                      # trim to batch multiple
with open("gpt_X.csv","w") as fx, open("gpt_Y.csv","w") as fy:
    for x, y in rows:
        fx.write(",".join(map(str, x)) + "\n"); fy.write(f"{y}\n")
open("vocab.txt","w").write("\n".join(repr(c)[1:-1] for c in chars) + "\n")
print(f"V={len(chars)} S={S} windows={len(rows)}")
print("seed:", ";".join(map(str, idx[:S])))
