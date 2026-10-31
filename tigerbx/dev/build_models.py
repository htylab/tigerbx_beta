"""Rebuild the released ``_opt`` models from their exported (base) ONNX files and check them against the release.

Each released ``_opt`` model is its base export rewritten by ``tigerbx.dev.onnx_optimize`` with the options in
``BUILDS`` (the same options are recorded in the model's ``tigerbx_optimize`` metadata), plus the metadata entries
listed there. A rebuilt model's graph equals the released one; the file can differ in the ``tigerbx_optimized_from``
entry, which later versions of onnx_optimize add.

Usage::

    python -m tigerbx.dev.build_models                     # rebuild all into ./models_rebuilt and compare
    python -m tigerbx.dev.build_models out_dir --only TBX  # one model

Base files come from the model directory (``lib_tool.get_model``: the cache or TIGERBX_MODEL_DIR; they are kept
privately, not on the model hub). See
doc/model_build.md for how each base was exported. Requires ``onnx``.
"""

import argparse
import hashlib
import os

from tigerbx.core import models

HLC_HEADS = 'parc:0:57:argmax,lr:57:60:argmax,dw:60:63:argmax,ct:63:64:slice,cgw:64:67:slice'
TSEG_META = {
    'tigerbx_mode': 'aseg43',
    'labels': 'background + tigerbx label_all["aseg43"] (44 channels, logits)',
    'input': 'tBET brain, RAS, 1 mm, bet111 bbox crop (+16 voxel), float32 / max',
    'arch': 'U-Net, base 32 (32-64-128-256), 10.0M params',
    'released_in': 'tigerbx 0.3.0',
}

# released name -> (base export, onnx_optimize arguments, metadata entries set before the optimisation's own)
BUILDS = {
    'TBX': ('mprage_bet_v005_mixsynthv4.onnx', ['--labels'], {}),
    'TSEG': ('mprage_aseg43_v0011_allseg.onnx', ['--labels'], TSEG_META),
    'DGM': ('mprage_dgm12_v002_mix6.onnx', ['--labels'], {}),
    'HLC': ('mprage_hlc_v004_resunetplusT2.onnx', ['--heads', HLC_HEADS], {}),
    'GDM': ('vdm_unet3d_v002.onnx', [], {}),
}


def graph_sha256(path):
    import onnx
    return hashlib.sha256(onnx.load(path).graph.SerializeToString()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(key, out_dir, check=True):
    """Rebuild one released model (a BUILDS key) into out_dir; returns (path, graph equal to the release)."""
    import onnx
    from tigerbx import lib_tool
    from tigerbx.dev import onnx_optimize
    base, args, meta = BUILDS[key]
    name = getattr(models, key)
    out = os.path.join(out_dir, name)
    onnx_optimize.main([lib_tool.get_model(base), out, '--no-check'] + args)
    if meta:
        m = onnx.load(out)
        own = {p.key: p.value for p in m.metadata_props}
        del m.metadata_props[:]
        onnx_optimize._set_metadata(m, {**meta, **own})
        onnx.save(m, out)
    if not check:
        return out, None
    same = graph_sha256(out) == graph_sha256(lib_tool.get_model(name))
    print(f'{key}: {name} graph {"equals" if same else "DIFFERS from"} the release; file sha256 {file_sha256(out)} '
          f'(release {models.SHA256[name]})')
    return out, same


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out_dir', nargs='?', default='models_rebuilt')
    ap.add_argument('--only', nargs='+', choices=sorted(BUILDS), help='models to rebuild (default: all)')
    ap.add_argument('--no-check', action='store_true', help='do not compare with the released models')
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    bad = [k for k in (a.only or BUILDS) if build(k, a.out_dir, not a.no_check)[1] is False]
    if bad:
        raise SystemExit(f'differs from the release: {", ".join(bad)}')


if __name__ == '__main__':
    main()
