import os
import sys


_longclip_module = None
_longclip_tokenize = None

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CORE_DIR)
DEFAULT_LONGCLIP_ROOT = os.path.join(REPO_DIR, "third_party", "Long-CLIP")


def resolve_longclip_module(longclip_root=None):
    global _longclip_module, _longclip_tokenize
    if _longclip_module is not None:
        return _longclip_module, _longclip_tokenize

    last_exc = None

    try:
        import longclip as longclip_mod
        _longclip_module = longclip_mod
        _longclip_tokenize = longclip_mod.tokenize
        return _longclip_module, _longclip_tokenize
    except Exception as exc:
        last_exc = exc

    candidates = []
    if longclip_root:
        candidates.append(longclip_root)
    if os.path.isdir(DEFAULT_LONGCLIP_ROOT) and DEFAULT_LONGCLIP_ROOT not in candidates:
        candidates.append(DEFAULT_LONGCLIP_ROOT)

    for root in candidates:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from model import longclip as longclip_mod
            _longclip_module = longclip_mod
            _longclip_tokenize = longclip_mod.tokenize
            return _longclip_module, _longclip_tokenize
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"LongCLIP is not available. Install `longclip` or provide third_party/Long-CLIP "
        f"(default: {DEFAULT_LONGCLIP_ROOT})."
    ) from last_exc


def load_longclip_model(longclip_model_path, device, longclip_root=None):
    longclip_mod, _ = resolve_longclip_module(longclip_root)

    if not longclip_model_path:
        raise ValueError("LongCLIP model path is empty.")

    if not os.path.isabs(longclip_model_path):
        if not os.path.isfile(longclip_model_path):
            repo_relative = os.path.join(REPO_DIR, longclip_model_path)
            if os.path.isfile(repo_relative):
                longclip_model_path = repo_relative
            else:
                default_path = os.path.join(DEFAULT_LONGCLIP_ROOT, "checkpoints", "longclip-L.pt")
                if os.path.isfile(default_path):
                    longclip_model_path = default_path

    if not os.path.isfile(longclip_model_path):
        raise FileNotFoundError(f"LongCLIP model not found: {longclip_model_path}")

    model, preprocess = longclip_mod.load(longclip_model_path, device="cpu")
    model = model.to(device)
    model.eval()
    return model, preprocess
