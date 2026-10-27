"""Built-in canonical Sylvester Walsh-Hadamard sign vectors for Ternary-Bonsai-2-{2B,4B,9B,27B}."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Canonical bit-packed sign vectors (1 = +1.0, 0 = -1.0) extracted from Prism ML's
# Ternary-Bonsai-2-{2B, 4B, 9B, 27B} hadamard.json / GGUF prism.hadamard.sign_values.
BUILTIN_BONSAI2_SIGNS_B64: dict[int, str] = {
    2048: "GHwfBJcYpY+5ztGkc5q7aPjLWJXQOa/AAryPO5Y7anPuPNjhFWULtOfvQhD12BxPtTsRCFqEi/VNAvd7UEnmNAzMdtTmmNAx4OaQ2Kjxy3w4RWR+cKsH3cEypAwzwKf0sK85vSd7abjka/J31MP5x6KWyI+axo5cHRypWCaAgADRcJ9aAZ/sy58L66LWeahv0yzFgz4cn8PKYnQLTdAM393MF4awO2Gd7XRyAipsFTf2m7lb9eWADAOGbU2dIXCF1WfjTr+9t4HoSMjPiJ5hgAKh63eT0mTt0HWNvFMCuu4FhJshDmxoupwCyDgjAkMzGdicguWu0KThv8paFzWffQ==",
    2560: "HMLahT0ZiUTTEYOVxiYXE7NQ8qEIVGvK2XkDO8AeLrQcQ82MNrib1ZDDwIUV5h60OYiwbsyzRj1lB/uZjR53VlYpDcFoZ9HKVKDMiGJ9ZRhLCGb4FgjkOGTpX4BxZtwwCsINRJhT0AWv2D8pU4J4Blm+n5h4r77Sn3js6Li4DQW8qPtqNfsRYxAiDLxy0HPAsgArwfEifBFO+e29ccbaDlNtVseqbGyCmgGTTDsG6cbEIbSANF/fE5RK92s0yiBie40HNPoDrCj7uv7pCWzbGjjgC/iK3Ec6l7PFa4DUnXWozac2S3HGlXIFl+/eu9InE4yXqaVKQoUofi0tTIIUnfa0ZqrtGa8ohcfMmPREGEdwnDHlb4vy7Y2FlvGLwTeHqL76XyV7jyc3TGlMn0y0zer+qHVXuixdDt5n9RUdq5Y=",
    4096: "GVeG8338zo6imrGJAaZcyZzOePtINyG9mxMJNhAtqCk5yMvZWGCfp1tWDCZ6ziByWGKfYdNQJxzgJaKF7/+MN8wYo97tareOnc01ZU/ko9ZRERIaqucDpgSSbQj0EvqdrK6FJGJ+f3Y/TqNt1lrboYJ0sazywWsWhEH5p7dCPRyJgTGTaBobtpmcEfKOCy/tUoG9U5XP4Bbfm6I75Muay5A9+pBDGhMzDUtf8SyeGM828NBwm4mSGAVMLhZCcG6qh8s8doNdxX7Ct9Gk5DhEs7iS/r2KZo77ESbIPzWx09x3MG3blsEFfbrRdx2SUErGbdGvYC0NCUCuYWIMkgd2kyf6s9a6W39yXQRNhyzLL6Vxd2A80xDMj9VsBDz/mafHjdKG7FERi/Q9MEFd5bAaIAFioRc5obB+4UYmYlegzex6wj6E0/0iec77pmymS2560K+Owa/3QzPqT5AZTSNVtGAZd/5IfGfOg/et0oZ+vUHV12MuAu/XsaVyzdfGH6A4TrzXpC4NB+K3cTG92JXfm3510XdjuuJ6cDULNeA+HZ6Jrlfe6FUz1zSyoy4oiUmChEc3wXb3/Lrq6vMbZPr3NB1npm94Ew7D9AlHE2dhtC0XUYptPP5WpQOMBG/T2Yzhiu1EVmizEqv/+ym805tNQbr6WGfXjZBTBXWhIGHh6ms=",
    5120: "F/ARQvDt2rfKatfY7lvAChxAb8KNNcYoeXGMVxF0rVj++wrSwKmcdOlF2t/S0ZibMMYYSrWamGem//USfuNQO3MI9oI3u+hRmbdaBLolCJW218AgW6uVTt3Yik4uivM8cRxzJkKYsMlwVYx4bsvQCQZopdFILvXvkrOq3MIrBY3iBk5qiImHcyWTUmh3B8PdDM2nas2cMZZ4kM1/Je0MHuiTCLsLxN5iUTmZsyhb0MyU9ghVhze8GDNohxBaoHBhx7s0j/OJ/Bb7fRzELANBbUcBlD2VoazXXqGkYJqVpoLItL0GoSvdpkEf3nKZcquqInpg8yr9fxF+rJTb7SOMl8Y1BcQydzI+bghOzrvJF0iJZnIoB3XJ0tEFbvdXgsCIuWdGeBQTeDCmAxkoAGiznxVAic1BOAjSJC9e8AfHINAis93bEQpgYjq/rHWk8uCyek9ADEIgXmO2O8U7129jrmgnyxwXbp2TLWlxNo1/DnSX4gsnogCN0LLtVfIwPzdg68T1Uj06YhO6kiowYjBreTHYZs6HQQw4W9BYjGF8SiyOy9T9ZkeHDwgt0ljBlcyzZTMsJzZgQWyMjp35k2k5t6Fpwf2n6mMArtyXG4oMGYmZMMZCgn9vA3GSg1tso0mC4pNqoQ2rm54NTuVY7y7soWdCPPW+I/GGSCr7ZMOrHiYXIOcnpSEKWUEwcZlzvzW+MLHQUTJeWibuNT8sbKOchpyzoJ65RgSbsFfTMnKWd3+W0TY1se63V6eXu+9TU0CgqgqeFaH/5wi8zFLXMgJ/3MNMF+GGJtcgBMynNtjwI6DhMd7Pe2IjwpFxZk9FvUMuDB0puGNRZINxLSzi+Xhu4Q==",
    6144: "P8SY5d+YCRdw4fVEGCd0zAYTP/XzjeuELNP04/ov9EWXdS17/TTg7gN76wo2FLaQNDzZcV0yr8JkACmsemOlHqVVAGRVx7vaiegZczlP1a7XOynvEqxgLkJwGWX9yUy/HGbsnHwuZzFE3jUFGl9qjVYfL4hZEbil2pUtJiUykrmasl6d6oRA52jqG2E1aphs7uibfKaItSnjBh+GYgfWADmKUpL4Tmn6s+bGdC+ZLItkfd6Q1AULQa9YR71Akx5LJG6AYPxYUrXxO4NKO7eWmk+Nz91+pkizdXJD3psWCkHpbH2NjvEHgFUc0GIq+FcBQGWoAmLQoM2MsGsQs4yUvlTEBBbq9p2zu3ZEBrDVVHqmupwMArayxStXNZlb92/xaOsOlh3ZRO0X/+Wg+XxRAFNjGxCQjEQLKdwRQQS4GleCSVnIO0hgBpPzC1b5z5ttMRLeTXGhI4uFyRFm0uNC0+IyX437XxXjwn4rtJAF+5keQuoDcldjf6NxSnV5a588agOGaWmSETBDOkxAsQjnY6FbzvtBXLJ5WlONijl0XEYff7fTVz5UlszVN41c9LwcFbzWSw8YEfFII4M4THCdumgNyl5CTyrJPVbNUoKAKaPdYK5Rjf5jQJyh3rOPdhk8viGY4OvNv9RGMTtgDUkReeL4Jw3RrR+6C8/JKv+nxBJG1dfbjrZRfFsbByWE5MxAjJk4zp8D7bGafQ3qRaMuF3Sfan0hqqiCah0m0NC1+JXmJLaN/ZYfjg/VNX5WyFF2aCXeUvflga0Kotr300ZnbGNpXa0+Zalqc92ccE7/PXWPN6KC6vc456TJRwram92406SzvbQP/ICNATSrAaKFsS/OV8cEUAu7hpE0DChp7a73voNClJuGk3CJWcc7pUP3XCoMxHRPqem6pqsRGkACjOVMn+sAOy42Jeyz3bCiIPFiWgLffjtM9eUOTbQO44RFV5KbqYTVoiGwPESpaGzRmOQR7ME+7t9YiCsJBIr7bRJRku+wuTUKPRn3lyP2EK5w",
    9216: "ffv7Z9JTCtxWKAt8Hi3O7zHmY73Hpi5eEklSem+lO0xpn5+mFaEOTpohjPjaSTcHw6kQXRLxds6rrcz7ekMUph6uDsFcM99MuaQ4mShnDxczyN79Z7lWfPBL/8UcQYKeQdgNHJgFg6OiR/BYhVokkZeF/JgoDOd6QtbLawIt24y1z3JmshZZ+z5AIj81Jl1HnctPEEfyZcx3WAYHfVJcRiYNqZ8XoHll3PJYHeOFbS8pAu+/9B3gRSr71IyUcEn/yV8OGODq+zbw+dddtE4Xf3Ce2IJPgpM0gIn7Sd7iQElUc8+MO6nZcmUCUx0uzCiNvY6gv9ia95wCmIve6IrlHQKlCz6/oJgzSqsv9cv43SpNVU7A+FogD+ba8Zattuz5u4YT0iZVeTzaX+CKT9p0MF40n6KLMBSqXo+9yMcdEKEU7ByYJ0GTkGhRAhp9CVtyvp48Mc75DyUiSmvCFQFjsJ3lbtkPEvfl/sj8FJU2ddtHezHZCy1OPkyEg4CGpQkltNLpHIQ0OYDeYXSNyBZWPVC+UydWD47G5IN03b7LqQkCAaaP0YUJkzkHDcZXIOz8Te86W6YJdmfVHA76E3Cuk2svfXlp5A2czgHGRAsP8zfbiPu1cr7eV7hsvqFOOopbyeelv5ZI2rwqD3hX+fUHYTfO8Gs90jKOhlmV2SP43ArZzaE/extFi34i8zaqEZJjCY5wx4wiKHsyLkMy/CiwAwo/ZOvfyKdqNIasEs5HXAsmBK8rdtHaPRayvKuMrAyRp1bir1DSIRMsGNw2Fzy7T9OV4vfrTJFWz5ZLnGTuCSmCO2tLgCP+jRY4VjF++R17mBc52kU2Vi/VkKHZYevOLJkU1tYh52gf8iWniYexS7ip3touZUMr/wVZAPEByoIUEOi9q00UM2PwV3GudKHRFd9sP1dYW1PXQhYUNL7jncQtobSdwbum8zuYSvtUw+2PN3CwQY6+SXKNc95+EaQ3IPKvV+voL8hOkdFW4bLWQm7/7W3NczWX880T473IQ/nf02Y3q8YLw60b41fU4JVDquLXnYcrxJNcobE2I3recl8bPS+7RFKnu43H27VLRgiKzq9TF7I/nc7p0CwBmyHC17xqL8b0ElBbNQOMGHGaVNLWHyNB2xYBeZI8PX5rfkw4ciSJou/w7JQFePUNfeANacnzxDLmlnOmZyn/ObG/ky+9MLDISZQYy2Jr//uSC22SCnm5nFSOtwRKp5KoqMyR1Hggp7Kz/TuXFpWTupNju8oVIvI3olZfttJbfLwcJvGgKC0C/38VMpa5avhquXvdBikYHYonqXe4yYI2wPMI9LwWJsIRQ0q7RFWaa2qfmNvmURJ9NlePYUWm7i5nIkNNhG+BXAIPCZWZS1Yz9yd6ZtzI85wU04t2jFCeG5qZMMunfMVA/97xL3DLKwu3HlVaohcmgljL2kHshF1vlwN37XQwgsPL6LPHyGYtvSMQuJ70rRxJpK+kU+0JCc27JIsSQfi84HeZ5Mre2MxzRoSYxMXcBZk1HUwbBIrfV2ELRFeF",
    12288: "9MrJgueA2Qusvb7oy+Yg1lyBGNTvW22RtkuJPiWDxVFNAofDn2ak8S5yOAUhUS33EDScIM+uMgIl2TMdhP9uJI6xbvuYEnAeEooZDDpEFO9INIntoihrUmMK5XLOS1ke9QnAKz5Nu24zHkXl4FLq4zS598etdyWFjFbMeusKvrxsXek/fEM4c8Uy1rowq18haSWNEqfFLIPlVAHwLatU5dUcg53x/3KaSxUz8RAw7dLSAoi5QVQgJqvnNcOXGc+Y/Xil6ANQx/n+yp5lDF9L9XcLnoAfT9aJb24up7erQ209eqMEEhZrauS/0v/v7XmjUuhz5V2Q/FOJJkiUbqJrNrge5dBGVQ0NzDKxeMWz6Cp6Zn4nwkQC0PrR1hj1AGrGAO9WKD+EzPUTXv6lp4/GE8FX9a4JR6USG0HGonrl4vQ6fW2Ft0xRMucvJngucA5n/1QBDNGrjJbOvXuadK1cUnyGA+QKL4NPwjCRJZEoFT+ylQFjRCa/wQvOzoAfMYDE4XLx6vV+TkE55lEUHXkGRCNATnq2e3o5Yj4P7vfUxoSWr5yTEDPp6f/CnB/cheTTGximWYATNqeAFo67HS1bTxEayFTwGNR1GfCfp9sAiqXJbx6w1DSYeqeIQJdB9Qn1qNFV/0rtgnrOjYLCXpVbQpbaI/88fBcJKNjPlaHx7CSFi1313jn6h+YIByeH4iBjj0tJqThnFoqLx5Simv8aDfLJVdKqTVCPsxDrt1H6erX9ZEUDbnCx7QIgWRBSA/10fIBI2DOXUXy/Fv0b+4WNKIhYdcl8VqZNoqVNKC8oNICiETdH7EDjtZPSXbrJmIhbGYUA7ZEnTC/MhPhIqQ5a41+vnN6VQbG76M5W41RuZEhiXOUVMxFrt6JUjm2SFQMJeLkgLa1ySEy9vPVYFvX31xuGgkshW6YJJ9DFVjW1rWLzIF4HP+IJJ/O5UFMRRi77KF+75iavSayhgPm4PvQJnrb6FT3KsSETnsHiPvQk2FmEbaioJeEA61dUtHqivjzunQ906BCwn7K9WUjfn0eNFMAIrFK+cfVQjAeMu0f8gEjhz7J+8L0JhpFQba5IqjXc2ZPwyoJ/+xtT9X88swHUolBr5BdfJ63EYGyzXHZkNLgh9/rlyrWul6QBFR1SUFF+DB4ksTv/OhUp6YCX/gL9IACyhnhPE1SSm7TJ0aCgRsv9yayhcF55j0sZpu68ZKmjcmAT8pfnbAyyN6hb5ED4Yw4S4vRvGzLDVy2ygFVt62Bv9rxz3vifAuQVon+cIqCjk8V1ihFDIXlTiFblL8n8Tds3wOEe2CgXWVatquI1i8qq0hRb6hA9aeBL/Z+BuQppwkYLduuY2VOkZPXi+Dp6IY1KVAdYBGIbbwBmSedXM2X94gJCLhVwxLF2xJieAlNg8vmc2c2p8LVSfYSjrdLkg/J6bo+1kuXRCu5o1i59dC9iUdWD9GO2Eq5LdqOKWeXitrk7nQ3cMkWZXi+rplXydUawsz8hzEbwJq9gTTycYag+j7nQ80XKMaFnqoNd0XL2cfg7/qppbp5jTgGGEnDuh+bEreSBWcS3MdoVInb89t5L2Fhg2f480t389J0C5q9pTzeH4VlqIWXKIoWiMw2BFmsfTfkgoKwZa/B4LYwUPu+j4gkAygXaBPee2v6o0sIVLpUeP8MCkN2/SQ05i8QS5gOPC8ZfhZtbHmi6l+tL8LwPpohjDQmx2h7ayaAPKDMA5ObNw58JxEFRcDrFivJK3+ICv9hndGYaetICKDJvZKk7CuugyTLM6/5/4nLrU1jpLZJ5/BfAYiMvMu1xKM1rn+DjyEetnedz/or2Q0fKLZNJIY730a0227ZiRyW0tFnu082OYMoLphqSRpWaiy0ShY5GPPt8hlg0/lF1O8/7EmXTpRz0EVKzdkzHdKY+pNosAj+zrhlaUcmq+UaRDRKoxZ0n3NGgbSfzH24pbHCVWqCgjh9LzUBsah1Zfi5pWFtKcAt9xDnq99+0BZdSiGipQLUzUf0B3+k2AH+8MmREZ/PaO646iqNCV/Cjm3OBnp1g",
    17408: "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",
}

# Map Qwen3.5 hidden_size -> canonical Bonsai-2 Hadamard block_size
BONSAI2_HIDDEN_TO_BLOCK_SIZE: dict[int, int] = {
    2048: 512,   # Ternary-Bonsai-2-2B
    2560: 512,   # Ternary-Bonsai-2-4B
    4096: 4096,  # Ternary-Bonsai-2-9B
    5120: 1024,  # Ternary-Bonsai-2-27B
}


def _unpack_b64_signs(width: int, b64_str: str) -> torch.Tensor:
    """Unpack bit-packed base64 string into a float32 tensor of {-1.0, +1.0} of length `width`."""
    raw = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
    bits = np.unpackbits(raw)[:width]
    signs = np.where(bits == 1, 1.0, -1.0).astype(np.float32)
    return torch.from_numpy(signs)


def load_signs_table(signs_path: str | Path | None = None) -> dict[int, torch.Tensor]:
    """Load a mapping `{width: signs_tensor}` from built-in table, hadamard.json, MLX dir, or GGUF."""
    table: dict[int, torch.Tensor] = {
        w: _unpack_b64_signs(w, b64) for w, b64 in BUILTIN_BONSAI2_SIGNS_B64.items()
    }
    if not signs_path or str(signs_path).lower() == 'builtin':
        return table

    p = Path(signs_path)
    if p.is_dir():
        if (p / 'hadamard.json').exists():
            p = p / 'hadamard.json'
        elif (p / 'model.safetensors').exists():
            from safetensors import safe_open
            with safe_open(str(p / 'model.safetensors'), framework='pt', device='cpu') as f:
                for k in f.keys():
                    if k.endswith('.signs'):
                        t = f.get_tensor(k).to(torch.float32).flatten()
                        table[int(t.numel())] = t
            return table

    if p.suffix == '.json' and p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        widths = [int(x) for x in data['prism.hadamard.sign_widths']]
        vals = data['prism.hadamard.sign_values']
        off = 0
        for w in widths:
            table[w] = torch.tensor(vals[off : off + w], dtype=torch.float32)
            off += w
        return table

    if p.suffix == '.gguf' and p.exists():
        import gguf
        reader = gguf.GGUFReader(str(p))
        fields = {k: v.contents() for k, v in reader.fields.items()}
        widths = [int(x) for x in fields['prism.hadamard.sign_widths']]
        vals = fields['prism.hadamard.sign_values']
        off = 0
        for w in widths:
            table[w] = torch.tensor(vals[off : off + w], dtype=torch.float32)
            off += w
        return table

    raise FileNotFoundError(f'Unsupported or missing bs2_qat_signs_path: {signs_path}')


def get_signs_for_width(
    width: int,
    signs_table: dict[int, torch.Tensor] | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return canonical +/-1 sign vector for `width`, falling back to deterministic seed."""
    if signs_table is None:
        signs_table = load_signs_table('builtin')
    if width in signs_table:
        return signs_table[width].to(device=device, dtype=torch.float32).clone()
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed) + int(width))
    bits = torch.randint(0, 2, (width,), generator=gen, dtype=torch.int64)
    signs = torch.where(bits == 1, 1.0, -1.0).to(dtype=torch.float32)
    return signs.to(device=device)


def infer_bonsai2_block_size(model: nn.Module, default: int = 512) -> int:
    """Infer canonical Hadamard block_size (512, 1024, or 4096) from model config."""
    cfg: Any = getattr(model, 'config', None)
    hidden_size: int | None = None
    if cfg is not None:
        text_cfg = getattr(cfg, 'text_config', None)
        if text_cfg is not None:
            hidden_size = getattr(text_cfg, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(cfg, 'hidden_size', None)
    if hidden_size in BONSAI2_HIDDEN_TO_BLOCK_SIZE:
        return BONSAI2_HIDDEN_TO_BLOCK_SIZE[int(hidden_size)]
    return default
