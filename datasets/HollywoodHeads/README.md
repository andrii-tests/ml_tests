# HollywoodHeads dataset of human head annotations in Hollywood movies

Created by Anton Osokin and Tuan-Hung Vu at INRIA, Paris.

### Introduction
HollywoodHeads dataset contains 369,846 human heads annotated in 224,740 video frames from 21 Hollywood movies. The movies vary in genres and represent different time epochs. The dataset is divided into the training, validation and test subsets which have no overlap in terms of movies. In brief, the training set of HollywoodHeads contains 216,719 frames from 15 movies, the validation set contains 6,719 frames from 3 movies and the test set contains 1,302 frames from another set of 3 movies. Human heads with poor visibility (e.g., strong occlusions, low lighting conditions) were marked by the “difficult” flag and were excluded from the evaluation.

### License

HollywoodHeads dataset is released under the MIT License (refer to the LICENSE file for details).

### Citing 

If you find our dataset useful in your research, please consider citing our paper:

    @inproceedings{vu15heads,
        Author = {Vu, Tuan{-}Hung and Osokin, Anton and Laptev, Ivan},
        Title = {Context-aware {CNNs} for person head detection},
        Booktitle = {International Conference on Computer Vision ({ICCV})},
        Year = {2015} }

### Contents
1. [Dowload](#download)
2. [Dataset structure](#dataset-structure)
3. [Data demo](#data-demo)
4. [Evaluation demo](#evaluation-demo)

### Download
Download and unpack the dataset
  ```Shell
  wget http://www.di.ens.fr/willow/research/headdetection/HollywoodHeads.zip
  unzip HollywoodHeads.zip
  ```

### Dataset structure
The dataset is organized as follows:
  ```Shell
  $HollywoodHeads/JPEGImages               # video frames
  $HollywoodHeads/Annotations              # VOC-style XML annotations
  $HollywoodHeads/Splits                   # train, validation, test splits
  ```

### Data demo
To run the demo code run the following file from MATLAB:

  ```Matlab
  cd HollywoodHeads
  demo_load_GT
  ```

### Evaluation demo
For the evaluation demo see our [code](https://github.com/aosokin/cnn_head_detection)