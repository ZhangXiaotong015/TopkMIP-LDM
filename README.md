# TopkMIP-LDM
The official implementation of [Top-K Maximum Intensity Projection Priors for 3D Liver Vessel Segmentation](https://arxiv.org/pdf/2503.03367v1.pdf).

## Data preparation
`python data_prepare.py`
## System matrix preparation
`python system_matrix.py`
## KL auto encoder training
`bash run_ircadb_trainAE.sh`
## Conditioning latent diffusion model training
`bash run_Abla1st_ircadb_trainLDiff.sh`
## Inference
`bash run_Abla1st_ircadb_inference.sh`
## Vessel tree reconstruction
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
