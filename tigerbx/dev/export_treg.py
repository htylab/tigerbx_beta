"""Export tReg v3 (deformable template registration) from its PyTorch weights to ONNX, and check it.

    python -m tigerbx.dev.export_treg weights.pth mprage_treg_v003.onnx [--compare other.onnx]

The network is defined here (no outside dependencies):

* features: one tBX U-Net shared by both images (the siamese part). Encoder levels (Conv3d 3x3x3 -> GroupNorm(8) ->
  ReLU) x 2 with MaxPool 2, widths 24-48-96-128-128; decoder 96-48-24-24 with skip connections. The features used
  for matching are the decoder outputs at full, 1/2, 1/4 and 1/8 resolution plus the 1/16 bottleneck, so even the
  finest level carries deep context.
* matching, coarse to fine (1/16 ... full): the moving features are warped by the current grid, the cosine similarity
  with the fixed features is computed for the 27 offsets in {-1, 0, 1}^3, a softmax with a learned temperature per
  level turns it into weights, the displacement is the weighted mean offset plus a small conv residual computed from
  the similarities, and it is composed as grid(p) <- grid_up(p + r(p)).

The weights are a state dict of ``network()`` (or a checkpoint holding it under ``'model'``). The ONNX model has the
same contract as the earlier tReg files: inputs moving / fixed [1, 1, 160, 224, 192] at any intensity scale
(min-max normalised inside), output grid = voxel sampling coordinates [1, 3, 160, 224, 192]
(moved(p) = moving(grid(p)), align_corners=True). ``--compare`` runs both models on the same random input pair.
Requires ``torch`` and ``onnx`` (neither is a TigerBx runtime dependency).
"""

import argparse

import numpy as np

SHAPE = (160, 224, 192)
META = {
    'tigerbx_mode': 'treg',
    'model': 'tReg v3 deformable template registration',
    'contract': 'inputs moving/fixed [1,1,160,224,192] any scale; output grid = voxel sampling coords',
    'arch': 'shared tBX U-Net features (24-48-96-128-128, decoder 96-48-24-24) + per-level 27-offset cosine '
            'matching with softmax, 3.6 M parameters',
}


def network():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def double_conv(cin, cout):
        return nn.Sequential(nn.Conv3d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                             nn.Conv3d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))

    def identity(x):
        return torch.stack(torch.meshgrid(*[torch.arange(n, device=x.device, dtype=torch.float32)
                                            for n in x.shape[2:]], indexing='ij'))[None]

    def sample(img, grid):
        n = torch.tensor(img.shape[2:], device=grid.device, dtype=grid.dtype).view(1, 3, 1, 1, 1)
        g = (grid / (n - 1) * 2 - 1).permute(0, 2, 3, 4, 1).flip(-1)
        return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

    def upsample_grid(grid, size):
        n = torch.tensor(grid.shape[2:], device=grid.device, dtype=grid.dtype)
        m = torch.tensor(size, device=grid.device, dtype=grid.dtype)
        up = F.interpolate(grid, size=size, mode='trilinear', align_corners=True)
        return up * ((m - 1) / (n - 1)).view(1, 3, 1, 1, 1)

    def correlation(f, m):
        f = F.normalize(f.float(), dim=1)
        m = F.pad(F.normalize(m.float(), dim=1), (1, 1, 1, 1, 1, 1))
        H, W, D = f.shape[2:]
        return torch.cat([(f * m[:, :, i:i + H, j:j + W, k:k + D]).sum(1, keepdim=True)
                          for i in range(3) for j in range(3) for k in range(3)], 1)

    offsets = torch.tensor([[i - 1, j - 1, k - 1] for i in range(3) for j in range(3) for k in range(3)],
                           dtype=torch.float32)

    class Encoder(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.levels = nn.ModuleList(double_conv(1 if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch)))

        def forward(self, x):
            out = []
            for i, lv in enumerate(self.levels):
                x = lv(x if i == 0 else F.max_pool3d(x, 2))
                out.append(x)
            return out

    class Features(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.encoder = Encoder(ch)
            dec = (ch[2], ch[1], ch[0], ch[0])
            self.dec = nn.ModuleList(double_conv((ch[4] if i == 0 else dec[i - 1]) + ch[3 - i], dec[i])
                                     for i in range(4))

        def forward(self, x):
            E = self.encoder(x)
            out = [E[4]]
            for i, d in enumerate(self.dec):
                up = F.interpolate(out[-1], size=E[3 - i].shape[2:], mode='trilinear', align_corners=True)
                out.append(d(torch.cat([up, E[3 - i]], 1)))
            return out[::-1]

    class TReg(nn.Module):
        def __init__(self, ch=(24, 48, 96, 128, 128), res_ch=16):
            super().__init__()
            self.encoder = Features(ch)
            self.log_beta = nn.Parameter(torch.zeros(len(ch)))
            self.res = nn.ModuleList(nn.Sequential(double_conv(27, res_ch), nn.Conv3d(res_ch, 3, 3, 1, 1))
                                     for _ in ch)

        def forward(self, moving, fixed):
            M, F_ = self.encoder(moving), self.encoder(fixed)
            grid = None
            for lv in range(len(M) - 1, -1, -1):
                grid = identity(M[lv]) if grid is None else upsample_grid(grid, list(M[lv].shape[2:]))
                c = correlation(F_[lv], sample(M[lv], grid))
                w = torch.softmax(c * self.log_beta[lv].exp(), 1)
                r = torch.einsum('bnxyz,nd->bdxyz', w, offsets.to(w.device)) + self.res[lv](c)
                grid = sample(grid, identity(r) + r)
            return grid

    return TReg()


def export(weights, out):
    import onnx
    import torch

    sd = torch.load(weights, map_location='cpu', weights_only=False)
    net = network()
    net.load_state_dict(sd.get('model', sd))
    net.eval()

    def n01(x):
        lo, hi = x.amin(), x.amax()
        return (x - lo) / torch.clamp(hi - lo, min=1e-6)

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, moving, fixed):
            return self.net(n01(moving), n01(fixed))

    x = torch.rand(1, 1, *SHAPE)
    with torch.no_grad():
        torch.onnx.export(Wrap(net).eval(), (x, x), out, input_names=['moving', 'fixed'], output_names=['grid'],
                          opset_version=20, dynamo=False)
    m = onnx.load(out)
    for nd in m.graph.node:                    # negative Transpose perms from the tracer -> positive
        if nd.op_type == 'Transpose':
            for a in nd.attribute:
                if a.name == 'perm' and any(v < 0 for v in a.ints):
                    vals = [v % len(a.ints) for v in a.ints]
                    del a.ints[:]
                    a.ints.extend(vals)
    for k, v in META.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(m)
    onnx.save(m, out)


def _run(path, moving, fixed):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return s.run(None, {'moving': moving, 'fixed': fixed})[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('weights')
    ap.add_argument('out')
    ap.add_argument('--compare', help='another ONNX export to compare outputs with')
    a = ap.parse_args(argv)
    export(a.weights, a.out)
    print('exported', a.out)
    if a.compare:
        rng = np.random.default_rng(0)
        moving, fixed = (rng.random((1, 1) + SHAPE, dtype=np.float32) for _ in range(2))
        d = np.abs(_run(a.out, moving, fixed) - _run(a.compare, moving, fixed))
        print(f'max |grid difference| {d.max():.2e} voxels, mean {d.mean():.2e}')


if __name__ == '__main__':
    main()
