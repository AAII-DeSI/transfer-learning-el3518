# MSCLDA
The implementation of "Multi-Source Contribution Learning for Domain Adaptation" in Python. 

Code for the TNNLS publication. The full paper can be found [here](https://doi.org/10.1109/TNNLS.2021.3069982).
## Contribution

- Development of a new method to learn weights of source domains using their predicted pseudo labels of target domain. 
- A representation extraction framework to explore the similarities and the diversities among all source and target domains, which enriches transfer information by providing multiple views of common and specific features. 
- An alignment structure to learn the similarities between source and target domains by measuring domain-level and class-level discrepancies simultaneously, which undermines the misalignment of boundary samples. 

## Overview
![Framework](https://github.com/AAII-DeSI/transfer-learning-el3518/blob/main/TNNLS-2021-MSCLDA/image/flowchart.jpg)

## Setup
Ensure that you have Python 3.7.4 and PyTorch 1.1.0

## Dataset
You can find the datasets [here](https://github.com/jindongwang/transferlearning/tree/master/data).

## Usage

Run "offh.py" for dataset OfficeHome, "off31.py" for dataset Office-31. 

## Results

| Task  | D | W  | A | Avg  | 
| ---- | ---- | ---- | ---- | ---- |
| MSCLDA  | 99.8  | 98.8  | 73.7  | 90.8 |



Please consider citing if you find this helpful or use this code for your research.

Citation
```
@article{li2021multi,
  title={Multi-Source Contribution Learning for Domain Adaptation},
  author={Li, Keqiuyin and Lu, Jie and Zuo, Hua and Zhang, Guangquan},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
