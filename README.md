# Food Vision 101 🍔👁
**Food-101:** A Delicious Deep Learning Journey

![Food Vision 101 Logo](/docs/images/food-101.png)

Food Vision 101 is a machine learning project that aims to classify and recognize different types of food items using deep learning techniques. This project provides a pre-trained model that can identify various dishes and ingredients from images.

A goal of this project: 
* beat the **[Food101 PAPER](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)** with only 10% of the data, a 2014 paper with average accuracy of 50.76%. 
* beat the **[DeepFood PAPER](https://www.researchgate.net/publication/304163308_DeepFood_Deep_Learning-Based_Food_Image_Recognition_for_Computer-Aided_Dietary_Assessment)** with 100% of the data, a 2016 paper which used a Convolutional Neural Network trained for 2-3 days to achieve 77.4% top-1 accuracy.

> 🔑 **Note:** 
> * **Top-1 accuracy** means "accuracy for the top softmax activation value output by the model" (because softmax ouputs a value for every class, but top-1 means only the highest one is evaluated). 
> * **Top-5 accuracy** means "accuracy for the top 5 softmax activation values output by the model", in other words, did the true label appear in the top 5 activation values? Top-5 accuracy scores are usually noticeably higher than top-1.

|  | 🍔👁 Food Vision 100% DATA  | 🍔👁 Food Vision 10% DATA |
|-----|-----|-----|
| Dataset source | TensorFlow Datasets | Preprocessed download from Kaggle | 
| Train data | 75,750 images | 7,575 images | 
| Test data | 25,250 images | 25,250 images | 
| Mixed precision | Yes | Yes |
| Data augmentatin | Yes | Yes |
| Data loading | Performanant tf.data API | TensorFlow pre-built function |  
| Target results | 77.4% top-1 accuracy (beat [DeepFood paper](https://arxiv.org/abs/1606.05675)) | 50.76% top-1 accuracy (beat [Food101 paper](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf)) | 

*Table comparing difference between Food Vision full data versus Food Vision mini (10% of the data).*

This repository uses methods to significantly improve the speed of model training:
1. Prefetching
2. Mixed precision training

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [License](#license)
- [Results](#results)


## Installation

```bash
# Clone the repository
git clone https://github.com/Alex-Winner/food-vision-101.git

# Change into the project directory
cd food-vision-101

# Install the required dependencies
pip install -r requirements.txt
```

## Usage

Run the food recognition model on an image

```bash
cd src
python recognize_food.py --image images/pizza.jpg
```

## Model Training

|Model              | Loss  | Accuracy  |
|-------------------|:-----:|:---------:|
|Feature Extration  | 1.082 | 70.55%    |
|Fine Tuning        | 0.946 | 78.09%    |

### Learning

![](/docs/learning.png)

### Confusion matrix
![](/notebooks/confusion_matrix.png)

### F1 scores

![](/docs/f1_scores.png)


### Dataset Download script
To download dataset from Tensorflow data sets run `download_tfds.py` script
|Argument| Description|
|---|---|
|-h, --help           |show this help message and exit|
|--dataset DATASET    |Dataset to download|
|--data_dir DATA_DIR  |Directory to download the data|
|--shuffle            |Whether to shuffle the files during download|
|--split SPLIT        |Dataset splits to download, separated by commas|
|--no-download        |Do not download the dataset if it is already available locally|
|--verbose            |Increase output verbosity|

```bash
python download_tfds.py --dataset food101 --data_dir ./my_data --shuffle --split train,test --verbose
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
