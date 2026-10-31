"""Export the inspect model (tigerbx.inspect) from its PyTorch weights to ONNX, and check it.

    python -m tigerbx.dev.export_inspect model.pt inspect_v003.onnx [--compare released.onnx]

The network is defined here (it is small and has no outside dependencies); the weights are a state dict of
``Net(32, 96)``. The ONNX model has inputs x [n, 4, 96, 96, 96] and h [n, 11] and six outputs (the network's
template head is not exported); its metadata is the contract tigerbx.inspect reads. ``--compare`` runs both models
on the same random input. Requires ``torch`` and ``onnx`` (neither is a TigerBx runtime dependency).
"""

import argparse
import json

import numpy as np

CLASSES = ['T1w', 'T2w', 'FLAIR', 'b0', 'DWI', 'other']
N_HEADER, N_IN = 11, 4
OUTPUTS = ['contrast', 'stripped', 'epi', 'orientation', 'truncated', 'motion']
META = {
    'model': 'tigerbx image inspection (one 3D CNN, 3.6 M parameters, six outputs)',
    'contract': 'x [n,4,96,96,96] float32: 256 mm RAS cube at 2.67 mm on the intensity centre of mass (intensity / p99 '
                'clipped to 1.5, empty fraction, inside field of view, |image - 3x3x3 box mean| / p99 from the native '
                'grid clipped to 1.5); h [n,11] header features (feature 9 unused, 0); outputs logits',
    'contrast_classes': json.dumps(CLASSES),
    'orientation_classes': '24 = axis order index (6) x 4 + A-P flip x 2 + S-I flip; class 0 = as the header says',
    'motion': 'sigmoid = motion artifact score 0-1',
}


def network(width=32, res=96):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        """Stem (4 -> w; input 64 / 96 / 128 -> 32: kernel 4 / 5 / 8, stride 2 / 3 / 4), three stages of conv +
        strided conv, global mean; one head per task."""
        STEM = {64: (4, 2, 1), 96: (5, 3, 1), 128: (8, 4, 2)}

        def __init__(self, w, res):
            super().__init__()
            k, st, pd_ = self.STEM[res]
            layers = [nn.Conv3d(N_IN, w, k, stride=st, padding=pd_), nn.BatchNorm3d(w), nn.ReLU(inplace=True)]
            c0 = w
            for c in (2 * w, 4 * w, 8 * w):
                layers += [nn.Conv3d(c0, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True),
                           nn.Conv3d(c, c, 3, stride=2, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True)]
                c0 = c
            self.body = nn.Sequential(*layers)
            self.contrast = nn.Linear(c0, len(CLASSES))
            self.strip, self.template, self.epi = (nn.Sequential(nn.Linear(c0 + N_HEADER, 64), nn.ReLU(),
                                                                 nn.Linear(64, 1)) for _ in range(3))
            self.orient = nn.Linear(c0, 24)
            self.trunc, self.motion = (nn.Sequential(nn.Linear(c0, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(2))

        def forward(self, x, h):
            f = self.body(x).mean((2, 3, 4))
            fh = torch.cat([f, h], 1)
            return (self.contrast(f), self.strip(fh)[:, 0], self.epi(fh)[:, 0], self.orient(f), self.trunc(f)[:, 0],
                    self.motion(f)[:, 0])

    return Net(width, res)


def export(weights, out):
    import onnx
    import torch
    net = network()
    net.load_state_dict(torch.load(weights, map_location='cpu'))
    net.eval()
    x, h = torch.rand(2, N_IN, 96, 96, 96), torch.rand(2, N_HEADER)
    torch.onnx.export(net, (x, h), out, input_names=['x', 'h'], output_names=OUTPUTS, opset_version=17, dynamo=False,
                      dynamic_axes={k: {0: 'n'} for k in ['x', 'h'] + OUTPUTS})
    m = onnx.load(out)
    for k, v in META.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(m)
    onnx.save(m, out)
    with torch.no_grad():
        ref = [t.numpy() for t in net(x, h)]
    got = _run(out, x.numpy(), h.numpy())
    print(f'{out}: max |torch - onnx| {max(float(np.abs(a - b).max()) for a, b in zip(ref, got)):.3g}')


def _run(path, x, h):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(None, {'x': x, 'h': h})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('weights')
    ap.add_argument('output')
    ap.add_argument('--compare', help='a released inspect ONNX model to compare with')
    a = ap.parse_args(argv)
    export(a.weights, a.output)
    if a.compare:
        rng = np.random.default_rng(0)
        x, h = rng.random((2, N_IN, 96, 96, 96), np.float32), rng.random((2, N_HEADER), np.float32)
        d = max(float(np.abs(u - v).max()) for u, v in zip(_run(a.output, x, h), _run(a.compare, x, h)))
        print(f'max |output - {a.compare}| {d:.3g}')


if __name__ == '__main__':
    main()
