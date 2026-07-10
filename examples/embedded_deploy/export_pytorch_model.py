"""
export_pytorch_model.py — Export PyTorch models to safetensors for Dream.

Two modes:
  --train_demo              Train a toy MLP on synthetic patterns, export it.
  --input X --output Y      Convert an existing PyTorch .pth to safetensors.

Requirements:
  pip install numpy torch safetensors
"""
import argparse
import sys


def train_demo():
    """Train an MLP on synthetic digit-like patterns, export to safetensors."""
    try:
        import torch
        import torch.nn as nn
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"Missing: {e.name}")
        print("Run: pip install numpy torch safetensors")
        sys.exit(1)

    torch.manual_seed(42)

    # Generate synthetic "digit" patterns — 8x8 images where each digit class
    # has a unique template, with gaussian noise added.
    # This matches Dream's gen_digits() pattern so results are comparable.
    def make_dataset(n_per_class=100):
        templates = []
        for d in range(10):
            torch.manual_seed(d * 1000 + 1)  # deterministic templates
            t = (torch.randn(64) > 0).float()  # each class gets a unique binary mask
            templates.append(t)
        templates = torch.stack(templates)  # (10, 64)

        X_list, Y_list = [], []
        torch.manual_seed(42)
        for digit in range(10):
            base = templates[digit]
            for _ in range(n_per_class):
                noisy = base + 0.15 * torch.randn(64)
                X_list.append(noisy)
                Y_list.append(digit)
        X = torch.stack(X_list)
        Y = torch.tensor(Y_list, dtype=torch.long)

        # Shuffle
        perm = torch.randperm(X.shape[0])
        return X[perm], Y[perm]

    X, Y = make_dataset(100)
    print(f"Dataset: {X.shape[0]} samples, 64 features, 10 classes")

    # Simple 3-layer MLP — matches Dream's ai_training.dream architecture
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining PyTorch MLP...")
    for epoch in range(50):
        # Mini-batch training
        perm = torch.randperm(X.shape[0])
        total_loss = 0.0
        nb = 0
        for i in range(0, X.shape[0], 64):
            idx = perm[i:i+64]
            logits = model(X[idx])
            loss = loss_fn(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        if epoch % 10 == 0 or epoch == 49:
            with torch.no_grad():
                acc = (model(X).argmax(1) == Y).float().mean().item()
            print(f"  epoch {epoch:2d}  loss={total_loss/nb:.3f}  acc={acc:.3f}")

    # Final accuracy check
    with torch.no_grad():
        final_acc = (model(X).argmax(1) == Y).float().mean().item()
    if final_acc < 0.8:
        print(f"\nWARNING: Low training accuracy ({final_acc:.3f}). "
              f"Model may not have converged. Export anyway for format testing.")
    else:
        print(f"\nFinal accuracy: {final_acc:.3f}")

    # Save as safetensors
    state = model.state_dict()
    # Move to CPU and ensure contiguous (safetensors requirement)
    clean = {k: v.contiguous().cpu() for k, v in state.items()}
    save_file(clean, "pytorch_mlp.safetensors")
    print(f"\nSaved pytorch_mlp.safetensors")
    print(f"Tensors in file:")
    for name, tensor in clean.items():
        print(f"  {name}: shape {list(tensor.shape)}, dtype {tensor.dtype}")

    print(f"\nNext: build\\Debug\\DreamCompiler.exe examples\\inference_pytorch.dream")


def convert_file(input_path, output_path):
    """Convert any PyTorch .pth/.pt to safetensors."""
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"Missing: {e.name}")
        print("Run: pip install numpy torch safetensors")
        sys.exit(1)

    print(f"Loading {input_path}...")
    try:
        obj = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading: {e}")
        sys.exit(1)

    # Extract state_dict from various PyTorch save formats
    if hasattr(obj, 'state_dict'):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            state = obj['model']
        elif 'state_dict' in obj:
            state = obj['state_dict']
        else:
            state = obj
    else:
        print(f"Error: don't know how to extract tensors from {type(obj).__name__}")
        sys.exit(1)

    # Keep only actual tensors, make contiguous
    clean = {}
    skipped = 0
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.contiguous().cpu()
        else:
            skipped += 1
    if skipped:
        print(f"  ({skipped} non-tensor entries skipped)")

    save_file(clean, output_path)
    print(f"Saved {output_path}")
    print(f"Tensors ({len(clean)}):")
    for name, tensor in clean.items():
        print(f"  {name}: shape {list(tensor.shape)}, dtype {tensor.dtype}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export PyTorch model for Dream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python export_pytorch_model.py --train_demo
  python export_pytorch_model.py --input my_model.pth --output my_model.safetensors
""")
    parser.add_argument('--train_demo', action='store_true',
                        help="Train demo MLP on synthetic digit patterns, export it")
    parser.add_argument('--input', type=str, help="Path to PyTorch .pth file")
    parser.add_argument('--output', type=str, default='model.safetensors',
                        help="Output safetensors path (default: model.safetensors)")
    args = parser.parse_args()

    if args.train_demo:
        train_demo()
    elif args.input:
        convert_file(args.input, args.output)
    else:
        parser.print_help()
