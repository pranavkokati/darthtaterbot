#!/usr/bin/env python3
"""Train the Heretic NNUE:  (768 -> H) x 2 perspectives -> SCReLU -> 1  (pure numpy, CPU only).

Data: 40-byte PackedPos records produced by `heretic datagen` (self-play, search-labelled).
Target: t = lam * sigmoid(score * ln10/400) + (1 - lam) * game_result   (stm point of view)
Loss  : binary cross-entropy between sigmoid(net_out) and t.  Network output o satisfies
        win_prob = sigmoid(o)  <=>  centipawns = o * 400 / ln(10)  (= 173.7 * o).

Usage
  python3 python/train_nnue.py --data data/gen1.bin --hidden 128 --epochs 30 --out nets/h1.nnue
"""
import argparse, math, os, struct, sys, time
import numpy as np

REC = np.dtype([('sq', 'u1', 32), ('stm', 'u1'), ('flags', 'u1'), ('score', '<i2'),
                ('result', 'u1'), ('ply', 'u1'), ('pad', 'u1', 2)])
F = 768
LN10_400 = math.log(10) / 400.0


def load(paths, max_abs, keep_noisy=False, residual=False):
    arrs = [np.fromfile(p, dtype=REC) for p in paths]
    d = np.concatenate(arrs)
    if residual:  # handcrafted-eval sidecars written by `heretic hceeval <bin> <bin>.hce`
        hce = np.concatenate([np.fromfile(p + '.hce', dtype='<i2') for p in paths]).astype(np.float32)
        assert len(hce) == len(d), f"hce sidecar length {len(hce)} != records {len(d)} (re-run hceeval)"
    else:
        hce = np.zeros(len(d), dtype=np.float32)
    n0 = len(d)
    keep = (np.abs(d['score'].astype(np.int32)) <= max_abs) & ((d['flags'] & 2) == 0)
    if not keep_noisy:
        keep &= (d['flags'] & 1) == 0  # drop positions whose best move is a capture/promotion
    d = d[keep]
    hce = hce[keep]
    print(f"loaded {n0} records, kept {len(d)} quiet positions ({len(d)/max(n0,1)*100:.1f}%)", flush=True)
    return d, hce


def feature_indices(d):
    """Return int16 arrays [N,32] of active feature indices for the side-to-move and opponent
    perspectives, padded with 768 (a dummy column that is dropped)."""
    n = len(d)
    sq = d['sq']                                    # [N,32] bytes, two squares per byte
    codes = np.empty((n, 64), dtype=np.uint8)
    codes[:, 0::2] = sq & 15
    codes[:, 1::2] = sq >> 4
    stm = d['stm'].astype(np.int16)[:, None]        # 0 white, 1 black
    sqs = np.arange(64, dtype=np.int16)[None, :]
    pt = (codes & 7).astype(np.int16)
    col = (codes >> 3).astype(np.int16)
    present = codes != 0
    out = []
    for persp in (stm, 1 - stm):
        rel = (col != persp).astype(np.int16)
        flip = np.where(persp == 1, sqs ^ 56, sqs)
        idx = (rel * 6 + (pt - 1)) * 64 + flip
        idx = np.where(present, idx, F).astype(np.int16)
        # compact: sort so that real features come first, keep 32 columns
        idx.sort(axis=1)
        out.append(idx[:, :32].copy())
    return out[0], out[1]


def dense(idx):
    b = idx.shape[0]
    x = np.zeros((b, F + 1), dtype=np.float32)
    np.put_along_axis(x, idx.astype(np.int64), 1.0, axis=1)
    return x[:, :F]


def targets(d, lam):
    s = d['score'].astype(np.float32)
    p_search = 1.0 / (1.0 + np.exp(-s * LN10_400))
    p_res = d['result'].astype(np.float32) * 0.5
    return (lam * p_search + (1 - lam) * p_res).astype(np.float32)


class Net:
    def __init__(self, H, rng, residual=False):
        self.H = H
        self.residual = residual
        self.w1 = (rng.standard_normal((F, H)) * 0.05).astype(np.float32)
        self.b1 = np.full(H, 0.1, dtype=np.float32)
        self.w2 = (rng.standard_normal((2 * H, 1)) / math.sqrt(2 * H) * (0.05 if residual else 1.0)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def params(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, xs, xn):
        as_ = xs @ self.w1 + self.b1
        an = xn @ self.w1 + self.b1
        cs = np.clip(as_, 0, 1)
        cn = np.clip(an, 0, 1)
        hs, hn = cs * cs, cn * cn
        h = np.concatenate([hs, hn], axis=1)
        out = h @ self.w2 + self.b2
        return out[:, 0], (as_, an, cs, cn, h)

    def backward(self, xs, xn, cache, dout):
        as_, an, cs, cn, h = cache
        H = self.H
        dout = dout[:, None]
        gw2 = h.T @ dout
        gb2 = dout.sum(0)
        dh = dout @ self.w2.T                        # [B,2H]
        das = dh[:, :H] * 2 * cs * ((as_ > 0) & (as_ < 1))
        dan = dh[:, H:] * 2 * cn * ((an > 0) & (an < 1))
        gw1 = xs.T @ das + xn.T @ dan
        gb1 = das.sum(0) + dan.sum(0)
        return [gw1, gb1, gw2, gb2]

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            f.write(b'HNN2' if self.residual else b'HNN1')
            f.write(struct.pack('<ii', self.H, F))
            f.write(self.w1.astype('<f4').tobytes())
            f.write(self.b1.astype('<f4').tobytes())
            f.write(self.w2.astype('<f4').tobytes())
            f.write(self.b2.astype('<f4').tobytes())

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            magic = f.read(4)
            assert magic in (b'HNN1', b'HNN2')
            H, feat = struct.unpack('<ii', f.read(8))
            n = Net(H, np.random.default_rng(0), residual=(magic == b'HNN2'))
            n.w1 = np.frombuffer(f.read(F * H * 4), dtype='<f4').reshape(F, H).copy()
            n.b1 = np.frombuffer(f.read(H * 4), dtype='<f4').copy()
            n.w2 = np.frombuffer(f.read(2 * H * 4), dtype='<f4').reshape(2 * H, 1).copy()
            n.b2 = np.frombuffer(f.read(4), dtype='<f4').copy()
        return n


def bce(p, t):
    eps = 1e-7
    return float(-(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)).mean())


def evaluate(net, idx_s, idx_n, t, off, bs=32768):
    tot, mse, n = 0.0, 0.0, len(t)
    for i in range(0, n, bs):
        out, _ = net.forward(dense(idx_s[i:i+bs]), dense(idx_n[i:i+bs]))
        p = 1 / (1 + np.exp(-(out + off[i:i+bs])))
        tot += bce(p, t[i:i+bs]) * len(p)
        mse += float(((p - t[i:i+bs]) ** 2).sum())
    return tot / n, mse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--lr-final', type=float, default=5e-5)
    ap.add_argument('--lam', type=float, default=0.75, help='weight of search score vs game result')
    ap.add_argument('--max-abs-score', type=int, default=1800)
    ap.add_argument('--val-frac', type=float, default=0.02)
    ap.add_argument('--max-positions', type=int, default=0)
    ap.add_argument('--init', default='')
    ap.add_argument('--residual', action='store_true', help='learn a correction on top of the handcrafted eval (needs <data>.hce sidecars)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d, hce = load(a.data, a.max_abs_score, residual=a.residual)
    if a.max_positions and len(d) > a.max_positions:
        sel0 = rng.permutation(len(d))[:a.max_positions]
        d, hce = d[sel0], hce[sel0]
    off_all = (hce * LN10_400).astype(np.float32)
    t0 = time.time()
    idx_s, idx_n = feature_indices(d)
    t_all = targets(d, a.lam)
    print(f"features built in {time.time()-t0:.1f}s", flush=True)

    perm = rng.permutation(len(d))
    nv = max(1000, int(len(d) * a.val_frac))
    vi, ti = perm[:nv], perm[nv:]
    vs, vn, vt, voff = idx_s[vi], idx_n[vi], t_all[vi], off_all[vi]
    net = Net.load(a.init) if a.init else Net(a.hidden, rng, residual=a.residual)
    print(f"train {len(ti)}  val {nv}  hidden {net.H}", flush=True)

    m = [np.zeros_like(p) for p in net.params()]
    v = [np.zeros_like(p) for p in net.params()]
    b1c, b2c, eps = 0.9, 0.999, 1e-8
    step = 0
    steps_per_epoch = max(1, len(ti) // a.batch)
    total_steps = steps_per_epoch * a.epochs
    best = 1e9
    for ep in range(a.epochs):
        te = time.time()
        order = rng.permutation(ti)
        run = 0.0
        for bi in range(steps_per_epoch):
            sel = order[bi * a.batch:(bi + 1) * a.batch]
            xs, xn = dense(idx_s[sel]), dense(idx_n[sel])
            tb = t_all[sel]
            out, cache = net.forward(xs, xn)
            p = 1 / (1 + np.exp(-(out + off_all[sel])))
            run += bce(p, tb)
            grads = net.backward(xs, xn, cache, ((p - tb) / len(sel)).astype(np.float32))
            step += 1
            lr = a.lr_final + 0.5 * (a.lr - a.lr_final) * (1 + math.cos(math.pi * step / total_steps))
            for k, (pr, g) in enumerate(zip(net.params(), grads)):
                m[k] = b1c * m[k] + (1 - b1c) * g
                v[k] = b2c * v[k] + (1 - b2c) * g * g
                mh = m[k] / (1 - b1c ** step)
                vh = v[k] / (1 - b2c ** step)
                pr -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        vl, vmse = evaluate(net, vs, vn, vt, voff)
        # RMS error expressed in centipawn-equivalent via local slope at p=0.5 (~ 1/(4*LN10_400))
        rms_cp = math.sqrt(vmse) / (0.25 * LN10_400)
        print(f"epoch {ep+1:3d}/{a.epochs}  train bce {run/steps_per_epoch:.5f}  val bce {vl:.5f}  "
              f"val rmse(p) {math.sqrt(vmse):.4f} (~{rms_cp:.0f}cp)  lr {lr:.2e}  {time.time()-te:.0f}s", flush=True)
        if vl < best:
            best = vl
            net.save(a.out)
    print(f"best val bce {best:.5f}  -> {a.out}")


if __name__ == '__main__':
    main()
