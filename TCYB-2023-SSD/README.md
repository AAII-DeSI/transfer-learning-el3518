# SSD
The implementation of "Multidomain Adaptation With Sample and Source Distillation" in Python. 

Code for the TCYB publication. The full paper can be found [here](https://doi.org/10.1109/TCYB.2023.3236008).

## Contribution

- A two-step selective strategy to select transfer source samples and the dominant source domain.
- An enhancement mechanism to improve the performance across domains of source predictors by adapting pseudo-labeled and unlabeled target samples,
- A new combination rule that not only estimates the combination weights but also identifies the dominant source domain

## Overview
![Framework-Source](https://github.com/el3518/SSD/blob/main/img/flowchart_s.jpg)
![Framework-Adaptation](https://github.com/el3518/SSD/blob/main/img/flowchart_t.jpg)

## Setup
Ensure that you have Python 3.7.4 and PyTorch 1.1.0

## Dataset
You can find the datasets [here](https://github.com/jindongwang/transferlearning/tree/master/data).

## Usage
Taking Office-31 as example:

First, run "off31_presf.py" to train the source model.

Then, run "off31_select.py" to select transferable samples.

Next, run "off31_domain.py" to rank the importance of source domains and identify the dominant domain.

Last, run "off31.py" to adapt source and target domain based on the selected samples and domain(s).

## Results

| Task  | D | W  | A | Avg  | 
| ---- | ---- | ---- | ---- | ---- |
| SSD  | 99.8  | 99.1  | 76.0  | 91.6 |


Please consider citing if you find this helpful or use this code for your research.

Citation
```
@article{li2023multidomain,
  title={Multidomain Adaptation With Sample and Source Distillation},
  author={Li, Keqiuyin and Lu, Jie and Zuo, Hua and Zhang, Guangquan},
  journal={IEEE Transactions on Cybernetics},
  year={2023},
  publisher={IEEE}
}
