#!/usr/bin/env python3
"""Check the closed-form Singer propagation used by _coast_predict.

The filter coasts open-loop across the multi-second observation gaps this stack
sees in the field.  The old form advanced position by 0.5*a*dt^2 while decaying
the acceleration it reported, so the two disagreed and the quadratic term threw
the estimate metres past the target.  These forms are the exact integral of a
Singer acceleration a(t) = a0 * exp(-t/tau), so the contribution saturates.
"""
import numpy as np

TAU = 0.5


def coast(p0, v0, a0, dt, tau=TAU):
    """Same formulas as _coast_predict in triangulation_node.py."""
    alpha = np.exp(-dt / tau)
    a = a0 * alpha
    v = v0 + a0 * tau * (1.0 - alpha)
    p = p0 + v0 * dt + a0 * tau * (dt - tau * (1.0 - alpha))
    return p, v, a


def numeric(p0, v0, a0, dt, tau=TAU, n=200000):
    """Brute-force integration of the same model, as an independent reference."""
    t = np.linspace(0.0, dt, n)
    a = a0[None, :] * np.exp(-t / tau)[:, None]
    v = v0[None, :] + np.cumsum(a, axis=0) * (dt / (n - 1))
    p = p0[None, :] + np.cumsum(v, axis=0) * (dt / (n - 1))
    return p[-1], v[-1], a[-1]


def main():
    p0 = np.array([1.0, -2.0, 3.0])
    v0 = np.array([2.0, 0.5, -1.0])
    a0 = np.array([1.5, -0.8, 0.4])

    for dt in (0.05, 0.2, 1.0, 3.0):
        p, v, a = coast(p0, v0, a0, dt)
        pn, vn, an = numeric(p0, v0, a0, dt)
        assert np.allclose(p, pn, atol=2e-3), 'posizione a dt=%s: %s vs %s' % (dt, p, pn)
        assert np.allclose(v, vn, atol=2e-3), 'velocita a dt=%s: %s vs %s' % (dt, v, vn)
        assert np.allclose(a, an, atol=1e-9), 'accelerazione a dt=%s' % dt

    # A decaying acceleration cannot contribute more than a0*tau to the velocity,
    # however long the gap.  The old form grew without bound.
    _, v_long, _ = coast(p0, v0, a0, 30.0)
    assert np.allclose(v_long, v0 + a0 * TAU, atol=1e-6), v_long
    v_old = v0 + a0 * 30.0
    assert np.linalg.norm(v_old - v0) > 10 * np.linalg.norm(v_long - v0)

    # Position drift over a 3 s gap: bounded here, quadratic before.
    p_new, _, _ = coast(p0, v0, a0, 3.0)
    p_old = p0 + v0 * 3.0 + 0.5 * a0 * 9.0
    assert np.linalg.norm(p_new - p_old) > 4.0, 'la vecchia forma dovrebbe divergere'

    # Zero acceleration must reduce to constant velocity exactly.
    p_cv, v_cv, _ = coast(p0, v0, np.zeros(3), 2.0)
    assert np.allclose(p_cv, p0 + v0 * 2.0) and np.allclose(v_cv, v0)

    print('ok: propagazione Singer coerente, contributo di accelerazione limitato')


if __name__ == '__main__':
    main()
