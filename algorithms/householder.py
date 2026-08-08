"""Householder QR wrapper for the standalone algorithms API."""

from __future__ import annotations

from typing import Any

from black_diamond.solvers.linear.qr import qr_decompose as _qr_decompose


def householder_qr(a: Any) -> Any:
    """Compute the QR factorization via modified Gram-Schmidt (Householder-style)."""
    return _qr_decompose(a)
