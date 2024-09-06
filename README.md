# Artifact for the Paper On Practical Realization of Evasion Attacks for Industrial Control Systems at RICSS 2024

1. The data are available at: [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.13692004.svg)](https://doi.org/10.5281/zenodo.13692004)

2. place the downloaded `.zip` file in the project folder

3. Verify the integrity of the dataset by running the following command:

    ```bash
    md5sum dataset.zip
    ```

    expected output `b14c554293f5a936d43c98b92a023603  dataset.zip`

4.  execute

    ```bash
    python prepare_dataset.py
    ```
now you can reproduce the results of the paper by executing the notebooks contained inside the `defense_models` and `batadal_2_0_plots`

Those will produce the result in Table 3 and the plots in the appendix.

Project structure

```
Practical-Evasion-Attacks/
│
├── dataset #appear after executing prepare_dataset.py
│
├── defense_models/
│   └── C-Town
│       └── evaluation_paper.ipynb
│
├── batadal_2_0_plots/
│   └── new_dataset_physical_plots.ipynb
│       └── plots
│
├── prepare_dataset.py
│
└── README.md
```

The data were generated based on [DHALSIM v0.6.0](https://github.com/Critical-Infrastructure-Systems-Lab/DHALSIM/releases/tag/v0.6.0)

When using the code or the data please cite our work

```
@InProceedings{erba24practicalevasion,
author="Erba, Alessandro and Murillo, Andres F. and Taormina, Riccardo and Galelli, Stefano and Tippenhauer, Nils Ole",
title="On Practical Realization of Evasion Attacks for Industrial Control Systems",
booktitle="Proceedings of the 2024 Workshop on Re-design Industrial Control Systems with Security (RICSS '24)",
year="2024",
month=OCT
publisher="ACM",
doi={10.1145/3689930.3695213}
}
```