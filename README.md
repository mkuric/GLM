# The Gaussian Latent Machine: Efficient Prior and Posterior Sampling for Inverse Problems
This repository contains a complete source code to reproduce all experimental results from our paper [The Gaussian Latent Machine: Efficient Prior and Posterior Sampling for Inverse Problems](https://arxiv.org/abs/2505.12836), which has been recently accepted at the SIAM Journal on Imaging Sciences!

## Contents
This guide covers the following topics:
1. [How to Set Up an Environment to Run the Code?](#1-how-to-set-up-an-environment-to-run-the-code)
2. [How to Check If the Environment Was Set Up Correctly?](#2-how-to-check-if-the-environment-was-set-up-correctly)
3. [How to Run the Code?](#3-how-to-run-the-code)
4. [How to Reproduce All the Experimental Results from the Paper?](#4-how-to-reproduce-all-the-experimental-results-from-the-paper)
5. [How to Use the Gibbs Sampler for Inverse Problems?](#5-how-to-use-the-gibbs-sampler-for-inverse-problems)
6. [What Do These Miscellaneous Examples Demonstrate?](#6-what-do-these-miscellaneous-examples-demonstrate)
7. [How to Cite This Work?](#7-how-to-cite-this-work)

## 1. How to Set Up an Environment to Run the Code?
The code was developed and tested on Arch Linux using Python 3.12 and PyTorch 2.4 (CUDA 12.1). Some of the experiments are computationally intense and require a CUDA-capable GPU.

Please follow the steps below to set up an environment that can run our source code. 

1. Create and activate a new conda environment:
   ```sh
   conda create -n sampling python=3.12
   conda activate sampling
   ```

2. Install PyTorch with CUDA support:
   ```sh
   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install some standard Python dependencies via pip:
   ```sh
   pip install -r requirements.txt
   ```

4. Install our custom [`logsumexpv2`](https://github.com/zacmar/logsumexpv2) PyTorch CUDA extension that provides memory efficient log-sum-exp activation, its gradient, and some subroutines used by the Gibbs sampler. It is built from source, so the CUDA toolkit (`nvcc`) matching your PyTorch build and a C++ compiler have to be available:
   ```sh
   git clone https://github.com/zacmar/logsumexpv2.git
   cd logsumexpv2
   pip install .
   ```

5. You have to set the environment variable `GLM_DATA_DIR` if you want to reproduce the experiment results from the paper. You can do so on Linux via:
   ```sh
   export GLM_DATA_DIR="SOME DIRECTORY"
   ```
   You can add this command to your shell configuration (e.g., `.bashrc` or `.zshrc`) to have it set automatically.

[Optional] We use LaTeX to render text in our figures, which requires a LaTeX distribution together with `dvipng` and Ghostscript. For instance, you can install these on Arch via:

```sh
sudo pacman -S --needed texlive-basic texlive-latex texlive-latexrecommended texlive-latexextra texlive-fontsrecommended ghostscript
```

or on Debian/Ubuntu via:

```sh
sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super ghostscript
```

## 2. How to Check If the Environment Was Set Up Correctly?
With the `sampling` environment activated, follow the steps below to check that everything was set up correctly.

1. Run the following command to check that PyTorch was installed correctly and sees your GPU:
   ```sh
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   It should run without any errors and print `True`.

2. Run the following command from the root of the [`logsumexpv2`](https://github.com/zacmar/logsumexpv2) directory to check that the extension was built correctly:
   ```sh
   python test-multisigma.py
   ```
   It should run without any errors and print out some tensors.

3. Run the following command from the repository root to check that the environment variable `GLM_DATA_DIR` is set:
   ```sh
   python -m sampling.util
   ```
   It should run without any errors. It throws an `AssertionError` if the environment variable `GLM_DATA_DIR` is not set.

4. Run the following command from the repository root to check that LaTeX rendering works properly in the figures:
   ```sh
   python -m sampling.misc.latent_representations
   ```
   It should run without any errors and produce four figures. The x label in all figures should be properly rendered. Note that this script requires around one minute to finish.

## 3. How to Run the Code?
The source code is organized into the Python package `sampling`. At the top level, the `sampling` package contains two files -- `samplers.py` and `util.py`. These two files implement all sampling and helper routines, respectively. 

The `sampling` package is then further divided into four subpackages -- `examples`, `experiments`, `misc`, and `visualization`. These four subpackages contain all runnable code of interest. Each subpackage serves a different purposes:

| Subpackage    | Purpose                                                                                      |
|---------------|----------------------------------------------------------------------------------------------|
| experiments   | Reproduces all experimental results from the paper                                           |
| visualization | Visualizes all experimental results from the paper                                           |
| examples      | Contains examples that demonstrate how the Gibbs sampler can be used for inverse problems    |
| misc          | Contains some miscellaneous examples that numerically demonstrate some claims from the paper |

All scripts in the subpackages import from the `sampling` package, so they have to be run as modules from the repository root:
```sh
python -m sampling.[subpackage].[filename]
```
For instance, the command:
```sh
python -m sampling.misc.equilibration
```
runs the script `equilibration.py` from the `misc` subpackage. 

Trying to run a script directly by its path produces an error. For instance, running the alternative command from the repository root:
```sh
python sampling/misc/equilibration.py
```
fails with an `ModuleNotFoundError: No module named 'sampling'` error, because the repository root is in that case not on the module search path.

We will explain the purpose of every script in these subpackages and how to run them properly in the following three sections.

## 4. How to Reproduce All the Experimental Results from the Paper?
Code to reproduce all the experimental results from the paper is given in the `experiments` and `visualization` subpackages. 

The `experiments` subpackage contains runners to run the experiments and store the results in the directory specified by the `GLM_DATA_DIR` environment variable. The runners are started via:

```sh
python -m sampling.experiments.[filename]
```

The following table shows the correspondence between the [filename] of the runner and the subsection in the paper that contains the results:

| [filename].py    | Subsection                              |
| ---------------- | --------------------------------------- |
| baseline.py      | 4.1.2. Baseline Experiments             |
| priors.py        | 4.1.6. Image Prior Sampling Experiments |
| posteriors.py    | 4.2. Posterior Sampling Experiments     |

Note that some subexperiments in the prior and posterior sampling experiments require extremely long runtimes (~100 of hours). Therefore, the runners support optional arguments to specify which subexperiments to run. For instance, this can be used to distribute the computation onto multiple machines.  All available optional arguments can be found in the main function of the runners.

The `visualization` subpackage contains scripts that read the stored results and create the figures of the paper. They are started via:

```sh
python -m sampling.visualization.[filename]
```

| [filename].py                   | Subsection                              |
| ------------------------------- | --------------------------------------- |
| visualize_baseline_sampling.py  | 4.1.2. Baseline Experiments             |
| visualize_prior_sampling.py     | 4.1.6. Image Prior Sampling Experiments |
| visualize_posterior_sampling.py | 4.2. Posterior Sampling Experiments     |

## 5. How to Use the Gibbs Sampler for Inverse Problems?
The subpackage `examples` contains minimal starter code that shows how to use the Gibbs sampler with the GMM prior from the paper on your inverse problems. The examples are started via:

```sh
python -m sampling.examples.[filename]
```
Available examples are:
| [filename].py     | Inverse problem |
| ----------------- | --------------- |
| denoising.py      | Denoising       |
| deblurring.py     | Deblurring      |
| dct_inpainting.py | DCT inpainting  |

The starter code shows how to run the parallel chain approach from the paper. The runtimes of the examples can be shortened by switching to sequential collecting of samples and by reducing the number of iterations.

The provided prior was learned for 96 x 96 patches from the BSDS dataset. It might not work well for images of other sizes, or other types of images beyond natural images.

The provided prior requires no hyperparameter tuning of any kind when switching between inverse problems!

## 6. What Do These Miscellaneous Examples Demonstrate?
The subpackage `misc` contains some small miscellaneous examples that numerically check some of the claims made in the paper. 

They are not needed for reproducing the experimental results, nor for solving inverse problems. But they might be of interest to readers who want to check these claims for themselves, or gain some intuition. 

They run on CPU and can be started as described in [Section 3](#3-how-to-run-the-code) by running the command:

```sh
python -m sampling.misc.[filename]
```

All available miscellaneous examples with their corresponding [filename] and a description that explains what they demonstrate are given in the following table:
| [filename].py                    | What it demonstrates                                                                                                                                             |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| tie_breaking_examples.py         | Numerically demonstrates on a few Gaussian test cases of various dimensions that Proposition 2.6 is correct                                                      |
| equilibration.py                 | Demonstrates that the stochastic matrix-free equilibration algorithm of Diamond and Boyd performs similarly to vanilla diagonal preconditioning on a toy example |
| latent_representations.py        | Numerically demonstrates that the GSM representations of the Laplace and Student-t distribution given in Table 1 of the paper are correct                        |
| laplace_factor_approximations.py | Visualizes the two GMM parametrizations used in Subsection "4.1.5. Sensitivity to GMM Parametrization" for the Laplace factor                                    |

Note that the `latent_representations` example needs around one minute to finish.

## 7. How to Cite This Work?
The paper has been accepted for publication in the SIAM Journal on Imaging Sciences and will appear there soon. Until then, please cite the preprint if you find our work useful:
```bibtex
@article{Kuric:2025,
  title         = {{The Gaussian Latent Machine: Efficient Prior and Posterior Sampling for Inverse Problems}},
  author        = {Kuric, Muhamed and Zach, Martin and Habring, Andreas and Unser, Michael and Pock, Thomas},
  journal       = {arXiv preprint arXiv:2505.12836},
  year          = {2025},
  doi           = {10.48550/arXiv.2505.12836},
  url           = {https://arxiv.org/abs/2505.12836},
}
```

