# char-GPT in pure Dream

A 1-block character-level transformer (embedding + positional encoding +
Q/K/V projections + scaled-dot-product attention + residuals + FFN + head)
trained end-to-end in Dream with AdamW. Every op on the path is verified by
`gradcheck()` (44/44).

    python3 make_data.py            # corpus -> gpt_X.csv / gpt_Y.csv / vocab.txt
    dream run char_gpt.dream        # trains 8 epochs, then generates 160 chars
    dream run char_gpt.dream | python3 decode.py    # decoded text

Expected: loss 3.31 (= ln 28) -> ~3e-4; generation reproduces the corpus.
