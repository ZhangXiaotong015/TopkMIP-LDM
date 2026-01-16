# TopkMIP-LDM
The official implementation of [Top-K Maximum Intensity Projection Priors for 3D Liver Vessel Segmentation](https://ieeexplore.ieee.org/iel8/10980665/10980666/10980858.pdf).

## Dockerfile
You can simply build the inference image in a WSL2 environment using the Dockerfile in [Dockerfile/topkMIP](Dockerfile/topkMIP/).

Open PowerShell and enter the following command.
```
notepad $env:USERPROFILE\.wslconfig
```

Paste the following commands into the pop-up window.
```
[wsl2]
memory=96GB
processors=4
swap=0
localhostForwarding=true
```

Run the Dockerfile in WSL2.
```
cd Dockerfile/topkMIP
docker build -t topk_mip:latest .
## In run.sh, replace the src path in '--mount type=bind,src=/mnt/e/WSL/TestData/LiverVesselSeg/Pre-ablation/Portal,dst=/CT_root,readonly \' with your own data path.
bash run.sh
```
You can find the model weights at [this link](https://drive.google.com/drive/folders/1I8axZT0U4mUli0cDMlGoFD2R9QzRzUXJ?dmr=1&ec=wgc-drive-globalnav-goto) and download them to ```Dockerfile/topkMIP/model```.

For the complete workflow, the input is a liver-masked CT volume cropped to the liver region with a size of (256,256,slices), while the output is a binary mask of the liver vessels.

**This method does not work well for CT volumes with a low contrast-to-noise ratio (CNR). To handle low-CNR cases, please use our other method, [GATSegDiff](https://github.com/ZhangXiaotong015/GATSegDiff).*

```Contents of the output folder```

```/samples/sample_xxxxx_Seqxx.nii.gz:``` Partial-view latent integral projections (IPs) of the 3D vessel tree. (Output of the blue-colored latent diffusion model in the paper.)

```/samples/recon_xxxxx_Seqxx.nii.gz:``` Partial-view IPs of the 3D vessel tree. (Output of the green-colored auto-encoder in the paper.)

```/projections/Proj_xxxxx.nii.gz:``` Full-view (180 views) IPs of the 3D vessel tree.

```/recons/ReconFBP_xxxxx.nii.gz:``` FBP vessel reconstruction based on full-view IPs of the 3D vessel tree.

```/reconsOpt/ReconFBP_xxxxx_optIter_4.nii.gz:``` Optimized FBP vessel reconstruction at iteration 4 of the artifact suppression process. (The result corresponds to ```T``` in formula (4) in the paper.)

```/reconsOptBinaryPercent98:``` Binarization of the optimized FBP vessel reconstruction based on ```T >= percentile(T, p)``` in the paper. The value of ```p``` is recommended to be in the range ```[95, 98]```.

```/reconsOptNoiseCancel:```




## Apptainer/Singularity container system
If you have a Docker image built as mentioned above, you can save the Docker image to a ```.tar``` file and convert it to a ```SIF``` file, which is compatible with Apptainer.
```
docker save -o topk_mip.tar topk_mip:latest
```
You can use the bash file in [Apptainer](Apptainer/) to run the inference. 
```
cd Apptainer
bash bash_topkMIP.sh
```

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

