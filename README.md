# TopkMIP-LDM
The official implementation of 'Top-K Maximum Intensity Projection Priors for 3D Liver Vessel Segmentation'.

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
