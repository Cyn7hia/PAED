<div align="center"><img src="https://github.com/Cyn7hia/PAED/blob/main/image/Meta-VAE_sampler.png" height="300px"/></div>
<h2 align="center">PAED: Zero-Shot Persona Attribute Extraction in Dialogues</h2>

<div align="center">
    <a>
        <img alt="Python Versions" src="https://img.shields.io/badge/python-%3E%3D3.8-blue">
    </a>
    <a>
        <img alt="PyTorch Versions" src="https://img.shields.io/badge/pytorch-%3E%3D1.10.0-green">
    </a>
</div>

## Overview
🔥 This is a repository for our paper ([PAED: Zero-Shot Persona Attribute Extraction in Dialogues](https://aclanthology.org/2023.acl-long.544.pdf)) accepted in ACL'23 ([The 61st Annual Meeting of the Association for Computational Linguistics](https://2023.aclweb.org)). (Author list: Luyao Zhu, Wei Li, Rui Mao, Vlad Pandelea and Erik Cambria.)
>In this work, we develop a PAED dataset, PersonaExt, with 1,896 re-annotated triplets and 6,357 corrected utterance-triplet pairs. 
>We present a generation-based framework for zero-shot PAED. A novel HNS strategy, Meta-VAE sampler with CSC, is presented to enhance the performance of our model.
>Our model achieves better results than strong baselines in zero-shot PAED and negative sampling.

## Dataset
The dataset [PersonaExt](https://github.com/Cyn7hia/PAED/blob/main/data/ConvAI2/u2t_map_all.json) is publicly available, with 105 relation types.

## Requirements and Installation
* PyTorch >= 1.10.0
* Python version >= 3.8
* You may use the folowing instruction to intall the requirements.
```bash
pip install -r requirements.txt
```

## Usage
To run this code, please use the following command (take split `unseen_10_seed_0` as an example). 
>10 is the number of unseen labels; 0 is the random seed used for randomly selecting the 10 unseen labels.

```bash
python trainer_finetune.py
```
To change the number of test labels, you may find the variable `num_test_labels` in the 719th line in `trainer_finetune.py` and put any number from {5, 10, 15} into the list.
>For example, to change the number of test labels into 5, you may set:
```python
>>>num_test_labels=[5]
```

To change the random seeds for selecting the unseen labels, you may find the variable `seeds` in the 720th line in `trainer_finetune.py` and put any number from {0, 1, 2, 3, 4} into the list.
>For example, to change the seed into 1, you may set:
```python
>>>seeds=[1]
```

For users who runs the code for the first time, the variable `synthetic` of functoin `main` (in the 743th line of `trainer_finetune.py`) should always set as `synthetic=True`; You may set `synthetic=False` only if there has been a `synthetic.jsonl` file in the directory.

The default evaluation is tailored for the results of single triplet extraction. To get the evaluation for multiple triplet extraction for other dataset, e.g., fewrel, you may set the flag `mode='multi'` of the function `run_eval`. Additionally, you need to remember to place the target dataset under the directory of `/outputs/data/splits/zero_rte/[YOURDATA]/unseen_[x]_seed_[x]/`, which should conver `train.json`, `dev.json` and `test.json`.

## Issues and Usage Q&A
To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/Cyn7hia/PAED/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem.

## Citation
Please cite as
```bibtex

@inproceedings{zhu-etal-2023-paed,
    title = {{PAED}: Zero-Shot Persona Attribute Extraction in Dialogues},
    author = {Zhu, Luyao  and
      Li, Wei  and
      Mao, Rui  and
      Pandelea, Vlad  and
      Cambria, Erik},
    booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    month = {jul},
    year = {2023},
    address = {Toronto, Canada},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2023.acl-long.544},
    pages = {9771--9787}
}
```



