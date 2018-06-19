# Breast density classification with deep convolutional neural networks
## Introduction
This is an implementation of the model used for breast density classification as described in our paper ["Breast density classification with deep convolutional neural networks"](https://arxiv.org/pdf/1711.03674.pdf). The implementation allows users to get the breast density prediction by applying one of our pretrained models: a histogram-based model or a multi-view CNN. Both models act on screening mammography exams with four standard views. As a part of this repository, we provide a sample exam (in `images` directory). The models are implemented in both TensorFlow and PyTorch.

## Prerequisites

* Python (3.6)
* TensorFlow (1.5.0) or PyTorch (0.4.0)
* NumPy (1.14.3)
* SciPy (1.0.0)

## Data

To use one of the pretrained models, the input is required to consist of four images, one for each view (L-CC, L-MLO, R-CC, R-MLO). Each image has to have the size of 2600x2000 pixels. The images in the provided sample exam were already cropped to the correct size.

## How to run the code
Available options can be found at the bottom of the file or  `density_model_tf.py` or `density_model_torch.py`.

Run the following command to use the model.

```bash
python density_model_(torch|tf).py (histogram|cnn)
```
You should get the following outputs for the sample exam provided in the repository.

With `histogram`:
```
Density prediction:
        Almost entirely fatty (0):                      0.0819444
    	Scattered areas of fibroglandular density (1):  0.78304
        Heterogeneously dense (2):                      0.133503
    	Extremely dense (3):                            0.00151265
```

With `cnn`:
```
Density prediction:
        Almost entirely fatty (0):                      0.209689
        Scattered areas of fibroglandular density (1):  0.765076
        Heterogeneously dense (2):                      0.024949
        Extremely dense (3):                            0.000285853
```

The results should be identical for both TensorFlow and PyTorch implementations.

## Converting TensorFlow Models

This repository contains saved checkpoints of the original TensorFlow models. We include a script for converting from TensorFlow checkpoints to PyTorch pickles.

```bash
python convert_model.py \
    histogram \
    saved_models/BreastDensity_BaselineHistogramModel/model.ckpt \
    saved_models/BreastDensity_BaselineHistogramModel/pytorch_model.p

python convert_model.py \
    cnn \
    saved_models/BreastDensity_BaselineBreastModel/model.ckpt \
    saved_models/BreastDensity_BaselineBreastModel/pytorch_model.p
```

## Reference

If you found this code useful, please cite our paper:

**Breast density classification with deep convolutional neural networks**\
Nan Wu, Krzysztof J. Geras, Yiqiu Shen, Jingyi Su, S. Gene Kim, Eric Kim, Stacey Wolfson, Linda Moy, Kyunghyun Cho\
*ICASSP, 2018*

    @inproceedings{breast_density,
        title = {Breast density classification with deep convolutional neural networks},
        author = {Nan Wu and Krzysztof J. Geras and Yiqiu Shen and Jingyi Su and S. Gene Kim and Eric Kim and Stacey Wolfson and Linda Moy and Kyunghyun Cho},
        booktitle = {ICASSP},
        year = {2018}
    }
