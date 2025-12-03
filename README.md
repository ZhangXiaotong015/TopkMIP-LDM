# TopkMIP-LDM
The official implementation of [Top-K Maximum Intensity Projection Priors for 3D Liver Vessel Segmentation](https://ieeexplore.ieee.org/iel8/10980665/10980666/10980858.pdf).

## Data preparation
The [3D-IRCADb-01](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) dataset was used. To exclude the vena cava from the annotated vessel tree, we reannotated the liver masks and have released them in this repository.

Generate top-k MIP of CT scans and integral projections (IPs) of annotated vessel trees:

`python data_prepare.py`

## System matrix preparation
Projection matrix used for FBP reconstruction:

`python system_matrix.py`

## KL auto encoder training
`bash run_ircadb_trainAE.sh`

## Conditioning latent diffusion model training
Our method is built on the implementation of [latent-diffusion](https://github.com/CompVis/latent-diffusion).

`bash run_Abla1st_ircadb_trainLDiff.sh`

## Inference
Obtain the generated IPs of vessel trees:

`bash run_Abla1st_ircadb_inference.sh`

## Vessel tree reconstruction
FBP reconstruction based on the generated IPs:

`bash reconstruction_IRCADB.sh`

## Noise cancelling
`bash noiseCancel_IRCADB.sh`

## Physical resolution recovery
`python physical_resolution.py`

## Citation
If you use this work, please cite:
```bibtex
@inproceedings{zhang2025top,
  title={Top-K Maximum Intensity Projection Priors for 3D Liver Vessel Segmentation},
  author={Zhang, Xiaotong and Broersen, Alexander and Van Erp, Gonnie CM and Pintea, Silvia L and Dijkstra, Jouke},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement

This work was performed using the compute resources from the Academic Leiden Interdisciplinary Cluster Environment (ALICE) provided by Leiden University.

