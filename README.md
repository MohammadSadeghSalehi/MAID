# MAID: Method of Adaptive Inexact Descent for Bilevel Optimization  [![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2308.10098)

**[An Adaptively Inexact First-Order Method for Bilevel Optimization with Application to Hyperparameter Learning](https://arxiv.org/abs/2308.10098)**  
_Accepted to SIAM Journal on Mathematics of Data Science (SIMODS)_

This repository provides the official implementation of the algorithm proposed in the paper:

**An Adaptively Inexact First-Order Method for Bilevel Optimization with Application to Hyperparameter Learning**  
Mohammad Sadegh Salehi, Subhadip Mukherjee, Lindon Roberts, Matthias J. Ehrhardt  


MAID (Method of Adaptive Inexact Descent) is a first-order algorithm for bilevel optimization that adaptively adjusts both solution accuracy and step sizes. It is particularly well-suited to large-scale bilevel learning problems, such as hyperparameter optimisation in image processing and machine learning.

---

## 🔍 Overview

MAID solves bilevel problems with adaptive inexactness and includes a built-in line search mechanism to select suitable upper-level step sizes. This repository includes:

- A Python implementation of MAID
- Multiple example problems from machine learning and image processing
- Tools to reproduce key experiments from the paper

---

## 📁 Repository Structure

```text
.
├── MAID.py                  # Core MAID algorithm
├── MultiClassRegression.py  # Bilevel hyperparameter tuning on MNIST
├── Quadratic.py             # Synthetic bilevel problem (with analytical solution)
├── TV_denoising.py          # TV image denoising (parameter learning)
├── FoE.py                   # Field of Experts denoising (parameter learning)
├── Kodak_dataset/                    # Kodak dataset for image denoising
└── README.md                # This file
```

The repository is organized as follows:

- `MAID.py`: Contains the core implementation of the MAID algorithm.
- `MultiClassRegression.py`: Example demonstrating the application of MAID to learn regularization parameters for multinomial logistic regression on the MNIST dataset.
- `Quadratic.py`: A simple quadratic bilevel problem with an analytical solution, used for verification.
- `TV_denoising.py`: Example showcasing the use of MAID to learn parameters for Total Variation (TV) denoising on 2D grayscale images (Kodak dataset).
- `FoE.py`: Example demonstrating the application of MAID to learn parameters for Field of Experts (FoE) denoising on 2D grayscale images (Kodak dataset).
- `Kodak_dataset/`: Directory containing the Kodak dataset used in the denoising examples.
- `README.md`: This file, providing an overview of the repository.

## `MAID.py`: The MAID Algorithm Implementation

The `MAID.py` file implements the Method of Adaptive Inexact Descent (MAID) algorithm for solving bilevel optimization problems. The algorithm takes the following inputs:

- `theta`: Initial guess for the upper-level variable (hyperparameters).
- `x0`: Initial guess for the lower-level variable.
- `upper_level_obj`: An instance of a class defining the upper-level objective function. This class should have a method to evaluate the objective given the upper-level variable and the optimal lower-level solution.
- `lower_level_obj`: An instance of a class defining the lower-level objective function. 
- `eps`: Initial tolerance for the lower-level solver.
- `delta`: Initial tolerance for the conjugate gradient (CG) method.
- `beta`: Initial step size for the upper-level gradient descent.
- `rho`, `nu`: Factors for decreasing and increasing the step size during backtracking.
- `tau`, `nu`: Factors for decreasing and increasing the tolerances `eps` and `delta`.
- `eta`: Parameter for the inexact sufficient decrease condition.
- `maxBT`: Maximum number of backtracking iterations.

###Install the necessary dependencies:
To run any of the example scripts, navigate to the repository directory in your terminal and execute the corresponding Python file. Make sure you have the required Python libraries installed (e.g., NumPy, SciPy, scikit-learn, Pillow). You can install them using pip:
 ```bash
    pip install numpy matplotlib scikit-image torch tqdm
```
* The following dependencies are required: `torch`, `numpy`, `matplotlib`, `scikit-image`, `PIL`, `tqdm`.
  
## Examples

### `MultiClassRegression.py`

This script demonstrates how to use the MAID algorithm to learn the regularization parameters of a multinomial logistic regression model on the MNIST dataset. The upper-level objective is to maximize the generalization performance (evaluated on a validation set), while the lower-level problem is the training of the multinomial logistic regression model for a given set of regularization parameters.

To run this example, ensure you have the necessary libraries installed (e.g., PyTorch, scikit-learn, NumPy). Execute the script using:

```bash
python MultiClassRegression.py
```

### Quadratic.py

This script implements a simple bilevel optimization problem with quadratic upper and lower level objectives. Since the analytical solution can be computed for this problem, it serves as a useful tool for verifying the correctness and behavior of the MAID algorithm.

Run this example with:
```bash
python Quadratic.py
```

### TV_denoising.py

This example applies the MAID algorithm to learn the regularization parameter and smoothing parameter of Total Variation (TV) denoising for 2D grayscale images. The upper-level objective is to minimize the denoising error on clean images from the Kodak dataset, while the lower-level problem is the TV denoising process for a given set of parameters.

Before running, ensure the Kodak dataset is present in the data/ directory. Execute the script using:
```bash
python TV_denoising.py
```

### FoE.py

Similar to the TV denoising example, this script demonstrates the use of MAID to learn the parameters of a Field of Experts (FoE) model for image denoising. The upper-level objective is to find the FoE parameters that yield the best denoising performance on the Kodak dataset. The lower-level problem involves performing FoE denoising with a given set of parameters.

Ensure the Kodak dataset is in the data/ directory and run the script with:
```bash
python FoE.py
```


For the denoising examples, ensure the Kodak_dataset/ directory contains the Kodak dataset.

📚 Further Reading

For theoretical foundations, convergence analysis, and full experimental details, please refer to the original paper on arXiv:2308.10098.
