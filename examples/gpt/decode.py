# Decode generated indices (lines after ---GEN---) back to text.
import sys
vocab = [l.rstrip("\n") or " " for l in open("vocab.txt")]
lines = sys.stdin.read().splitlines()
idxs = [int(float(x)) for x in lines[lines.index("---GEN---")+1:]
        if x.strip().replace(".","").isdigit()]
print("".join(vocab[i] for i in idxs))
