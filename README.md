# MK-UNet Replication and Domain Generalization

Code accompanying:
- Step 1 execution summary and report (reference replication + domain generalization experiment on MK-UNet)
- Step 2 research proposal preliminary result (PRIME degree-centrality vs. segmentation difficulty)

Submitted to Prof. Md Mostafijur Rahman's lab screening process, Texas Tech University.

## Base repository

This project builds directly on top of the official MK-UNet implementation:
**https://github.com/SLDGroup/MK-UNet**

To reproduce, first clone that repository, then drop the files below into it
at the indicated paths (they assume the official repo's `mkunet_network.py`,
`utils/`, and directory layout are already present).

## Structure

```
step1_experiment1_clinicdb/
    sanity_check.py               architecture/params/FLOPs verification (no training)
    train_polyp_paperconfig.py    paper-text-config comparison run (patched copy of train_polyp.py)
    train_full_run.log            full log: shipped-defaults config, 5 runs x 200 epochs      
    train_paper_config_run.log    full log: paper-text config, 5 runs x 200 epochs         

step1_experiment2_brats/
    extract_brats_slices.py       BraTS2020 -> 2D axial FLAIR slice extraction pipeline
    train_brats.py                training script adapted for the extracted BraTS slices
    extraction_log.txt            slice extraction run log                                    
    patient_split.txt             saved patient-level train/val/test split assignment       
    train_brats_full_run.log      full log: BraTS shipped-defaults config, 5 runs x 200 epochs  

step2_preliminary_result/
    prelim_degree_vs_difficulty.py   degree-centrality vs. per-slice Dice correlation analysis
    prelim_result2_log.txt           run log                                                    
    prelim_result2_data.npz          raw per-slice dice/degree/correlation data                 
    prelim_result2_scatter.png       scatter plot figure                                        

environment/
    SimpleITK_stub.py             workaround for SimpleITK build failure under network isolation
```



## Environment notes

- Python 3.8, PyTorch 1.11.0+cu113, official repo's `requirements.txt` with two exclusions:
  `simpleitk` and `medpy` could not be built on a network-isolated server (SimpleITK's CMake
  build clones ITK from GitHub). `medpy` was installed with `--no-deps` (its actual usage here
  -- Dice/HD95/Jaccard/ASD metrics -- depends only on numpy/scipy). `SimpleITK` itself is stubbed
  (see `environment/SimpleITK_stub.py`) since it is imported unconditionally in the shared
  `utils/utils.py` but only used in unused `.nii.gz` volumetric save paths.
- Trained on 3x NVIDIA Titan V (12GB each).

## Data

- **CVC-ClinicDB**: obtained via the official repository's linked Google Drive folder (README).
  Authors' own pre-made 550 train / 62 test split used as-is.
- **BraTS2020**: obtained via a public Kaggle mirror (369 patients). 50 patients sampled
  (fixed seed), split at the patient level (40 train / 5 val / 5 test) -- see
  `patient_split.txt`. Not redistributed here; download separately.

## Full analysis

See the accompanying Step 1 report, Step 1 execution summary, and Step 2 research proposal
documents for full results, discussion, and interpretation. This repository contains the code
and raw logs referenced by those documents.
