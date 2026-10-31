# TigerBx: how the released models are built

Every default model in `tigerbx/core/models.py` can be traced to the step that produced it. The tools live in
`tigerbx/dev/` (they need `onnx`, and `torch` for an export; neither is a TigerBx runtime dependency).

| Model (`models.py`) | Released file | Built by | From |
|---|---|---|---|
| `TBX` | `mprage_tbx_v005_opt.onnx` | `python -m tigerbx.dev.build_models --only TBX` | `mprage_bet_v005_mixsynthv4.onnx` |
| `TSEG` | `mprage_tseg_v0011_opt.onnx` | `python -m tigerbx.dev.build_models --only TSEG` | `mprage_aseg43_v0011_allseg.onnx` |
| `DGM` | `mprage_dgm12_v002_opt.onnx` | `python -m tigerbx.dev.build_models --only DGM` | `mprage_dgm12_v002_mix6.onnx` |
| `HLC` | `mprage_hlc_v004_opt.onnx` | `python -m tigerbx.dev.build_models --only HLC` | `mprage_hlc_v004_resunetplusT2.onnx` |
| `GDM` | `vdm_unet3d_v002_opt.onnx` | `python -m tigerbx.dev.build_models --only GDM` | `vdm_unet3d_v002.onnx` |
| `INSPECT` | `inspect_v003.onnx` | `python -m tigerbx.dev.export_inspect model.pt inspect_v003.onnx` | PyTorch weights (state dict of `export_inspect.network()`) |
| `VFA` | `mprage_reg_v005_vfa.onnx` | the low-memory VFA export (outside this repository) | PyTorch checkpoint |
| `CT`, `VMNET`, `AFFINE_NET`, `EMBED` | as listed | earlier exports | — |

## `_opt` models

`tigerbx.dev.build_models` runs `tigerbx.dev.onnx_optimize` on each base export with the options the release used
(`--labels`; HLC `--heads parc:0:57:argmax,lr:57:60:argmax,dw:60:63:argmax,ct:63:64:slice,cgw:64:67:slice`; GDM none)
and adds the release's metadata entries. It then compares the rebuilt graph with the released model's: they are equal
for all five. The files' sha256 can differ in one metadata entry (`tigerbx_optimized_from`, added by later versions
of onnx_optimize), so the graph is what is compared. The options are also recorded in each model's
`tigerbx_optimize` metadata.

The base exports are not on the public model hub; they are kept privately (ask the maintainers). Put them in the
model directory (the cache, or `TIGERBX_MODEL_DIR`) before running `build_models`; their sha256 are in
`models.SHA256`, so `lib_tool.get_model` checks them like any release model.

## inspect

`tigerbx.dev.export_inspect` holds the network definition and the ONNX contract (metadata). Exporting the released
weights gives outputs identical to `inspect_v003.onnx`. The weights are kept privately with the base exports (ask
the maintainers).

## tReg

`mprage_treg_v001.onnx` and `mprage_treg_v002.onnx` are exported from their PyTorch checkpoints with a low-memory
matching step (the 27 neighbour similarities computed one shift at a time instead of a [27 x C] candidate tensor;
same output grid). The export needs a network source under GPL-3.0, which is therefore not part of this
repository. Neither file has been through `onnx_optimize`.

## A new model

See [developer_guide.md](developer_guide.md) §7: export, optimise, upload, add its sha256, and add a row here
(and, for an `_opt` model, an entry in `build_models.BUILDS`).
