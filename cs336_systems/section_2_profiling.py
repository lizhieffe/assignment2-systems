# Section 2: Profiling
#
# uv run cs336_systems/section_2_profiling.py

import torch
from cs336_basics.model import BasicsTransformerLM

BATCH_SIZE = 4
CONTEXT_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    lm = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=CONTEXT_LENGTH,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
    ).to(DEVICE)

    inp = torch.randint(0, 10_000, (BATCH_SIZE, CONTEXT_LENGTH)).to(DEVICE)
    out = lm(inp)
    print(out.shape)  # should be (BATCH_SIZE, CONTEXT_LENGTH, vocab_size)


if __name__ == "__main__":
    main()
