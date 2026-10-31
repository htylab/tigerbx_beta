"""The default model files and the sha256 of every model on the modelhub release.

One place for the names: the modules and pipelines take their defaults from here, `lib_tool.get_model()` checks a
cached or downloaded file against `SHA256`, and the executable's embedded models (`.github/workflows/exe.yml`
MODELS) must be listed here (tests/test_models.py). A new model: add its sha256, then change the default.
"""

TBX = 'mprage_tbx_v005_opt.onnx'
TSEG = 'mprage_tseg_v0011_opt.onnx'
DGM = 'mprage_dgm12_v002_opt.onnx'
CT = 'mprage_mix_ct.onnx'
HLC = 'mprage_hlc_v004_opt.onnx'
# tReg: the deformable template registration (reg plan step 'D', norm, template; embed uses v001's encoder).
# v001 = mprage_reg_v005_vfa (the same file; the full-resolution feature matching computes the 27 neighbour
# similarities one shift at a time: low peak memory), v002 = the same network trained for EPI as well
TREG = 'mprage_treg_v002.onnx'
VMNET = 'mprage_reg_v003_train.onnx'          # reg plan V
AFFINE_NET = 'mprage_affine_v002_train.onnx'  # affine_type 'C2FViT'
EMBED = 'mprage_embed_v001_vfa.onnx'
# _opt: v002 with staged GroupNorm statistics (onnx_optimize): faster, and the CPU gives the GPU's result
GDM = 'vdm_unet3d_v002_opt.onnx'
INSPECT = 'inspect_v003.onnx'

SHA256 = {
    # base exports of the _opt models (tigerbx.dev.build_models rebuilds the _opt files from them)
    'mprage_aseg43_v0011_allseg.onnx': 'c6163104e44277f8178575ebd62e097f8a02b40063176590b46d2581f7b6dfe1',
    'mprage_bet_v005_mixsynthv4.onnx': '7571ec093c0a54fa3a81f576063eaab0e3babac790f7946da00a6e23b4feda5d',
    'mprage_dgm12_v002_mix6.onnx': '2d9d5f09465f1883e6d0fcd4bf1972a450ecb6a49c033471c983b9f109f27d70',
    'mprage_hlc_v004_resunetplusT2.onnx': '00b56ad4fb3a008237216e07542b4b869c9e7003551d078c062437cc132b5ceb',
    'inspect_v003.onnx': 'afa33a88f814d6805152c2b0ffa6dc12f150ad0f1fc012203028935476c37b21',
    'mprage_affine_v002_train.onnx': '8912e93d8d8ff3c695ac44dabe10c8453787b74c390731d5ed34650d19148b5a',
    'mprage_dgm12_v002_opt.onnx': '77fb816a66ce4ae895f1ca044f34e3255fe4309cbc4d1c428fedf9f96a81a35e',
    'mprage_embed_v001_vfa.onnx': '833dd32a047238f147f4ae545ed3da32b97789d0cf8efed3300dc55650ef6c96',
    'mprage_hlc_v004_opt.onnx': '7358caadcfc8ed51e2293c5eb5e154bb611c3b4ad21e60be067eea0ea026ddd1',
    'mprage_mix_ct.onnx': '48affdf980a1039fe440b3df86dd76f23318bc49486dcf60d775a8a500037b96',
    'mprage_reg_v003_train.onnx': 'bf2ccf3f9328cf021fb1134b21a1b11d4a33cb9af4856060aa2640aa38d08525',
    'mprage_reg_v004_vfa.onnx': '6a4f4eada8853367919da657716bbbd1ce8b2d378126bd3f8c7770352bb85138',
    'mprage_reg_v005_vfa.onnx': 'b29c4c466e06064b35aa97ec24773aac08b19f16c8923da8cf6c53ae6217a013',
    'mprage_treg_v001.onnx': 'a47961e51a4a26142700d17b0b2dc60fdff2b2f9dd0dc3977d8bf54a3c849b33',
    'mprage_treg_v002.onnx': 'e6fd4edc5e6bae3312f557776acd181df1da65f0433f327e737d82c7513fbdb8',
    'mprage_tbx_v005_opt.onnx': '1fdb8ef2c36d1e2dc90e8987e4a9898d6399bb5f3b616d698bb5d192c8e5afa8',
    'mprage_tseg_v0011_opt.onnx': '67b7f553dfd99363f3f89268536986d24d1b79e9ce60760dc4e0ee969f755fe2',
    'vdm_unet3d_v002.onnx': '9357b40da479f4dc85efe2616cab02ad5ffe7f238221070a3769d767a968a97f',
    'vdm_unet3d_v002_opt.onnx': '230265338c5b9bf22bb0c38aeda2522969523368b02149f7ef441aa9c3005535',
}
