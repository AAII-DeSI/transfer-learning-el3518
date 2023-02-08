# DCA
The implementation of "Dynamic Classifier Alignment for Unsupervised Multi-Source Domain Adaptation" in Python. 

Code for the TKDE publication. The full paper can be found [here](https://doi.org/10.1109/TKDE.2022.3144423). 

## Contribution

- A new method to learn the importance of multi-view features and re-weight them to ensure the dominant features contribute more when merging their predictions.
- A self-training strategy to select pseudo target labels with high confidence in the training progress which improves the cross-domain ability of the source classifiers by iteratively splitting the target
domain into training and testing sets.
- An automatic sample-wise method to learn the weight vectors for conjoining multiple predictions from different views and source classifiers to estimate the target labels.

## Overview
![Framework](https://github.com/el3518/DCA/blob/main/image/flowchart-0.jpg)

## Setup
Ensure that you have Python 3.7.4 and PyTorch 1.1.0

## Dataset
You can find the datasets [here](https://github.com/jindongwang/transferlearning/tree/master/data).

## Usage
Run "dca-offh.py" to validate DCA on dataset Office-Home. 

## Results

| Task  | R | P  | C |  A | Avg  | 
| ---- | ---- | ---- | ---- | ---- | ---- |
| DCA  | 81.4  | 80.5  | 63.6 | 72.1 | 74.4 |


Please consider citing if you find this helpful or use this code for your research.

Citation
```
@article{li2022dynamic,
  title={Dynamic Classifier Alignment for Unsupervised Multi-Source Domain Adaptation},
  author={Li, Keqiuyin and Lu, Jie and Zuo, Hua and Zhang, Guangquan},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
