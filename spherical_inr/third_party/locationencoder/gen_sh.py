# =============================================================================
# spherical harmonics routines
# from: https://github.com/marccoru/locationencoder (MIT-licensed)
# =============================================================================


#!/usr/bin/env python3
"""
Simple generator for:
  - sh.py              (all the Yl*_m* + _SH_DISPATCH + SH)

Usage:
    python gen_sh.py
"""

import sys
from datetime import datetime
from sympy import assoc_legendre, cos, sin, sqrt, pi, factorial, Abs, Symbol

# ── CONFIGURE ──────────────────────────────────────────────────────
L = 30    # max ℓ
OUT = "."
# ────────────────────────────────────────────────────────────────────

theta = Symbol("theta")
phi   = Symbol("phi")

def calc_ylm(l, m):
    """Sympy expression for real Yₗᵐ(θ,φ) via the Wikipedia “real form”."""
    if m < 0:
        Plm = assoc_legendre(l, Abs(m), cos(theta))
        N   = sqrt((2*l+1)/(4*pi) * factorial(l-Abs(m))/factorial(l+Abs(m)))
        return (-1)**m * sqrt(2) * N * Plm * sin(Abs(m)*phi)
    elif m == 0:
        return sqrt((2*l+1)/(4*pi)) * assoc_legendre(l, 0, cos(theta))
    else:
        Plm = assoc_legendre(l, m, cos(theta))
        N   = sqrt((2*l+1)/(4*pi) * factorial(l-m)/factorial(l+m))
        return (-1)**m * sqrt(2) * N * Plm * cos(m*phi)

# 1) generate sh.py
with open(f"{OUT}/sh.py", "w") as out:
    out.write(f"""# =============================================================================
# Auto-generated real spherical harmonics (ℓ ≤ {L})
# From: https://github.com/marccoru/locationencoder  (MIT-licensed)
# Generated on: {datetime.now().date().isoformat()}
# =============================================================================

import torch
from torch import cos, sin

""")
    # emit each Yl*_m*
    for l in range(L+1):
        for m in range(-l, l+1):
            name = f"Yl{l}_m{m}".replace('-', '_minus_')
            expr = calc_ylm(l, m).evalf()
            out.write("@torch.jit.script\n")
            out.write(f"def {name}(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:\n")
            out.write(f"    return {expr}\n\n")

    # static dispatch dict
    out.write("_SH_DISPATCH = {\n")
    for l in range(L+1):
        for m in range(-l, l+1):
            fn = f"Yl{l}_m{m}".replace('-', '_minus_')
            out.write(f"    ({l},{m}): {fn},\n")
    out.write("}\n\n")

    # Python‐eager lookup
    out.write("""\
def SH(m: int, l: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    \"\"\"Lookup a real SH Y_l^m(theta, phi) via a static dict.\"\"\"
    fn = _SH_DISPATCH.get((l, m))
    if fn is None:
        raise ValueError(f\"Y_{l}^{m} not implemented (ℓ must be ≤ {L})\")
    return fn(theta, phi)
""")

print(f"→ wrote {OUT}/sh.py")

