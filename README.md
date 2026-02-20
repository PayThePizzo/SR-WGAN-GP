<div align="center">
    <h1>
        Super-Resolution of Medical MNIST Images Using Wasserstein Generative Adversarial Networks and Gradient Penalty (SR-WGAN-GP)
    </h1>
    <p>
        The goal of this project is to develop an interesting, although rudimental, generative model that is capable of
        upscaling to 64x64, images of medical scans (Breast MRIs, Chest X-Rays, ...) and chemical images, for their low-resoution version of 28x28. The challenge, a part from achieving stability in training, is to obtain a generator model capable of capturing small details (like ribs or knuckles) without losing the human-perspective of images.
    </p>
</div>

<br/>

# 📘 Table of Contents

- [📘 Table of Contents](#-table-of-contents)
  - [🌟 About the Project](#-about-the-project)
    - [👾 Tech Stack](#-tech-stack)
    - [🎯 Features and Roadmap](#-features-and-roadmap)
  - [🧰 Getting Started](#-getting-started)
    - [🔧 Configuration, Data Positioning and Environment Variables](#-configuration-data-positioning-and-environment-variables)
    - [❗ Prerequisites](#-prerequisites)
  - [👀 Usage](#-usage)
    - [🚀 Training](#-training)
    - [🔬 Testing](#-testing)
  - [💎 Acknowledgements and References](#-acknowledgements-and-references)

---

## 🌟 About the Project

The project follows the following structure:

- **Goal**: Build and train a Wasserstein GAN (WGAN) to enhance the resolution of Medical MNIST images. Specifically, the project will focus on generating higher-resolution (e.g., 64x64) versions of the original downsampled Medical MNIST images which are 28x28 pixels. This super-resolution technique should be able to show how WGANs can be applied to improve image quality, while leveraging the benefits of the Wasserstein distance to stabilize GAN training (along with GP).
- **Expected Implementation Steps**: The implementation should focus on developing and implementing a WGAN architecture, including both the Generator and Critic (the WGAN version of the Discriminator), the Wasserstein loss function with gradient penalty. This can also include pooling layers and/or optimizers like Adam (although they should not be the focus of this implementation). The technological stack I would like to use is the usual Python, Numpy, PyTorch (or tensorflow depending on which is easier to use with my GPU), Tensorboard (which allows tracking and visualizing metrics such as loss and accuracy), OpenCV (for image manipulation).
- **Model Evaluation**: The quality of the generated images could be evaluated using qualitative (visual inspection) and quantitative (metrics like Peak Signal-To-Noise Ratio, Structural Similarity Index Measure, etc...) methods.

### 👾 Tech Stack

<details>
    <summary>Core Languages and Libraries</summary>
        <ul>
            <li><a href="https://github.com/pyenv/pyenv">PyEnv 2.5.1</a> for Python version and virtualenv management</li>
            <li><a href="https://www.python.org/">Python 3.12.8</a> through PyEnv</li>
            <li><a href="https://python-poetry.org/">Poetry 1.8.5</a> for Python dependecy management</li>
            <li><a href="https://numpy.org/">NumPy 2.1.2</a></li>
            <li><a href="https://pytorch.org/">PyTorch 2.5.1</a> as the tensor library used to build the models</li>
            <li><a href="https://www.tensorflow.org/tensorboard">Tensorboard 2.18.0</a> and <a href="https://docs.pytorch.org/vision/stable/index.html">torchvision 0.20.1</a> to track the models' training</li>
            <li><a href="https://pypi.org/project/pillow/">pillow 11.0.0</a> to deal with I/O for images</li>
            <li><a href="https://docs.pydantic.dev/latest/"> Pydantic 2.11.5</a> to build automatically the running configuration</li>
            <li><a href="https://scikit-image.org/">Scikit-Image 0.25.0</a> to compute MSE, RMSE, SSIM, PSNR</li>
            <li><a href="https://github.com/mseitzer/pytorch-fid">pytorch-fid 0.3.0</a> to compute the FID score</li>
            <li><a href="https://pypi.org/project/lpips/">lpips 0.1.4</a> to compute the LPIPS score</li>
        </ul>
</details>

For more details please refer to the `pyproject.toml` file

### 🎯 Features and Roadmap

The following features have been implemented for the current project

- [X] Configuration
  - [X] Configuration parser module `/config/config_parser.py` that reads from `/config/config.yaml`
- [X] Logging
  - [X] Automatic logging through Tensorboard of losses, images and metrics during trainin
- [X] Input
  - [X] Type
    - [X] 3 Channel images (standard RGB)
  - [X] Loading with ad-hoc module
- [X] Preprocessing
  - [X] Normalization
  - [X] Grayscale
- [X] Generator Models
  - [X] Basic SR WGAN Generator
- [X] Critic Models
  - [X] WGAN critic with gradient penalty
- [X] Losses
  - [X] Generator
    - [X] Adversary loss
    - [X] Pixel-wise L1 loss
  - [X] Critic
    - [X] Wasserstein distance
    - [X] Gradient penalty
- [X] Training loop
  - [X] Load custom train and val sets
  - [X] Batch mode
  - [X] Validation phase
  - [X] Save model `.pth`
  - [X] Save a copy of model config
- [X] Test
  - [X] Load custom test set
  - [X] Save images (both real and fake)
  - [X] Compute metrics
  - [X] Compute FID of all test images
  - [X] Save .csv file of images perfomances

---

## 🧰 Getting Started

### 🔧 Configuration, Data Positioning and Environment Variables

Before doing anything make sure your project looks like this, or just create the missing folders yourself.

```txt
/
|- config
|- data/    # create if missing, this is where the MedicalMNIST data will be copied
|- logs/    # create if missing, this is where the logs for tensorboard are located
|- models/
|- runs /   # create if missing, this is where the results and model state will be saved
|- src/
|- .flake8
|- .gitignore
|- main.py
|- poetry.toml
|- Project Presentation Slides.pdf
|- Project Presentation.mp4
|- pyproject.toml
|- README.md
|- requirements.txt

```

Then to ensure we have the data (in case of using MedicalMNIST):

- You must retrieve the dataset from [here](https://www.kaggle.com/datasets/andrewmvd/medical-mnist/data) and unpack it
- Enter the main folder where the other subcategories are present (`BreastMRI`, `CXR`, ...)
- In any folder (we want to use) divide the images and put 90% of them into a new folder named `Train` and the rest into a folder named `Test`. Repeat this for the data of interest.
- Name the main folder with `MedicalMNIST` and copy it into data

This should look like this in the end:

```txt
/
|- config
|- data/    # create if missing, this is where the MedicalMNIST data will be copied
    |- MedicalMNIST/
        |- BreastMRI/
            |- Train/
                |- 000000.jpeg
                |- 000001.jpeg
                ....
            |- Test/
                |- 001000.jpeg
                |- 001001.jpeg
                ....
        |- CXR/ 
            |- Train/
                |- 000000.jpeg
                |- 000001.jpeg
                ....
            |- Test/
                |- 001000.jpeg
                |- 001001.jpeg
                ....
        ...
|- logs/    # create if missing, this is where the logs for tensorboard are located
|- models/
|- runs /   # create if missing, this is where the results and model state will be saved
|- src/
|- .flake8
|- .gitignore
|- main.py
|- poetry.toml
|- Project Presentation Slides.pdf
|- Project Presentation.mp4
|- pyproject.toml
|- README.md
|- requirements.txt

```

In case of using `40mic`, we have an alternative but similar pipeline.
This is needed for the data to be loaded. The very next thing to do is to check the `config.yaml` file:

```yaml
logging:
  interval: 5
  image_log_count: 16

data:
  dataset: "40mic"            # Folder to load the train set
  img_channels: 3
  train_percentage: 0.9
  batch_size: 16

preprocessing:              # For now they are ignored, but will be used in the future
  normalize: true
  normalize_mean: 0.5
  normalize_std: 0.5
  rotation: false         
  rotation_deg: 5
  rotation_p: 0.1   
  flip_horizontal: false  
  flip_h_p: 0.1
  flip_vertical: false    
  flip_v_p: 0.1
  gaussian_noise: false 
  noise_percentage: 0.1
  gaussian_p: 0.1

training:
  epochs: 50

validating:
  interval: 5

generator:
  mode: "classic"
  lr: 0.00001
  beta_1: 0.0
  beta_2: 0.9
  lambda_adv_loss: 0.5
  lambda_l1_loss: 0.2

critic:
  mode: "gp"                      # classic (weight clip) or gp
  lr: 0.00001
  beta_1: 0.0
  beta_2: 0.9
  lambda_wasserstrein: 1.0
  lambda_gp: 10
  weight_clip: 0.01           
  c_iter: 5
```

Now that the project is correctly configured we can move on. In the future we will add a CLI and a way to validate the configuration.

### ❗ Prerequisites

First of all install Python 3.12 (preferrably 3.12.8) and create a virtual environment. Then, activate the environment either use `requirements.txt` or Poetry 1.8.5 to install the dependencies.

```bash
# Poetry
(.venv) pip install poetry==1.8.5
# ...
(.venv) poetry install

# Or use pip
(.venv) pip install -r requirements.txt
```

Once the virtual environment is ready, if you are using a GPU, make sure it matches the version of the packages we use for PyTorch and related packages (or just use the cpu).

---

## 👀 Usage

We have two things we can do with our project: Train and Test

### 🚀 Training

After having determined the configuration, for which we can modify the `config.yaml` file, we can run our project with

```bash
# Train the model
python models/train.py
```

This will start the execution and create the following folders:

```txt
/
...
|- logs/
    |- [MODEL NAME]_[DATASET]_[TIME]/
        |- fake/
        |- metrics/
        |- real/
...
|- runs /  
    |- [MODEL NAME]_[DATASET]_[TIME]/
        |- config.yaml
...
```

Where `[MODEL NAME]_[DATASET]_[TIME]` is the name generated for the current run.

To see how the training is going we can just use tensorboard where everything is plotted for us:

```bash
tensorboard --logdir logs/[MODEL NAME]/

# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.19.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

Once the training is finished we find the model state in that folder

```txt
/
...
|- runs /  
    |- [MODEL NAME]/
        |- config.yaml
        |- model.pth    # model state
...
```

### 🔬 Testing

Now it is time to run the tests. We just need to find the folder in which out model state has been saved and run the following

```bash
#
python models/test.py runs/[MODEL NAME]/
```

This generates a `benchmark.csv` inside the folder, along with an `output/` folder where the generated and test images are copied to separated folders.
Thus, this results in the following thing:

```txt
/
...
|- runs /  
    |- [MODEL NAME]/
        |- output/
            |- generated/
                |- 00000.jpeg
                |- 00001.jpeg
                ....
            |- original/
                |- 00000.jpeg
                |- 00001.jpeg
                ....
        |- benchmark.csv
        |- config.yaml
        |- model.pth    # model state
...
```

And also prints a summary, containing the average scores, over the whole test set:

```bash
======== Test Set Evaluation ========
MSE: 0.0016
RMSE: 0.0398
PSNR: 28.1763
SSIM: 0.9245
LPIPS: 0.1074
Fid: 34.15489524220507
Saved metrics to: runs/wgan_gp_CXR_2025-06-21_18-38-03/benchmark.csv
```

---

## 💎 Acknowledgements and References

The following references have been the core foundations for us to develop the project:

- [1] C. Ledig, L. Theis, F. Huszar, J. Caballero, A. P. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi, "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," CoRR, vol. abs/1609.04802, 2016. [Online]. Available: http://arxiv.org/abs/1609.04802.
- [2] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, "Improved Training of Wasserstein GANs," CoRR, vol. abs/1704.00028, 2017. [Online]. Available: http://arxiv.org/abs/1704.00028
- [3] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein Generative Adversarial Networks," in Proceedings of the 34th International Conference on Machine Learning, vol. 70, D. Precup and Y. W. Teh, Eds., PMLR, 2017, pp. 214–223. [Online]. Available: https://proceedings.mlr.press/v70/arjovsky17a.html.
- [4] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative Adversarial Nets," in Advances in Neural Information Processing Systems, vol. 27, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Q. Weinberger, Eds. Curran Associates, Inc., 2014. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
- [5] A. Polanco, "Medical MNIST Classification," GitHub repository, 2017. [Online]. Available: https://www.kaggle.com/datasets/andrewmvd/medical-mnist/data 
- [6] V. Gupta, “Wasserstein GAN with Gradient Penalty (WGAN‑GP),” Medium, Jan. 19, 2025. [Online]. Available: https://medium.com/@vg498660/wasserstein-gan-with-gradient-penalty-wgan-gp-0892a51ea196

> Please feel free to contact me if anything is wrong, incorrect or not cited/used properly. I will immediately proceed to remove or modify anything that goes against copyright or any other guideline I might be ignoring.
