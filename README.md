<h2 align="center">PAED: Zero-Shot Persona Attribute Extraction in Dialogues</h2>

## Overview
🔥 &nbsp;This is a project for our paper ([PAED: Zero-Shot Persona Attribute Extraction in Dialogues](/paper/PAED.pdf)) accepted in ACL'23 ([The 61st Annual Meeting of the Association for Computational Linguistics](https://2023.aclweb.org)).
>In this work, we develop a PAED dataset, PersonaExt, with 1,896 re-annotated triplets and 6,357 corrected utterance-triplet pairs. 
>We present a generation-based framework for zero-shot PAED. A novel HNS strategy, Meta-VAE sampler with CSC, is presented to enhance the performance of our model.
>Our model achieves better results than strong baselines in zero-shot PAED and negative sampling.

## Requirements and Installation
* PyTorch >= 1.10.0
* Python version >= 3.8
* You may use the folowing instruction to intall the requirements.
```bash
pip install -r requirements.txt
```

## Uasge
To run this code, please use the following command (take split `unseen_10_seed_0` as an example). 
>10 is the number of unseen labels; 0 is the random seed used for randomly selecting the 10 unseen labels.

```bash
python trainer_finetune.py
```
To change the number of test labels, you may find the variable `num_test_labels` in the 717th line in `trainer_finetune.py` and put any number from {5, 10, 15} into the list.
>For example, to change the number of test labels into 5, you may set:
```python
>>>num_test_labels=[5]
```

To change the random seeds for selecting the unseen labels, you may find the variable `seeds` in the 718th line in `trainer_finetune.py` and put any number from {0, 1, 2, 3, 4} into the list.
>For example, to change the seed into 1, you may set:
```python
>>>seeds=[1]
```

The default evaluation is the results of single triplet extraction. To get the evaluation for multiple triplet extraction for other dataset, e.g., fewrel, you may set the flag `mode='multi'` of the function `run_eval`. Additionally, you need to remember to place the target dataset under the directory of `/outputs/data/splits/zero_rte/[YOURDATA]/unseen_[x]_seed_[x]/`, which should conver train.json, dev.json and test.json.

## Citation
Please cite as
```bibtex
@inproceedings{zhu2023paed,
  title = {PAED: Zero-Shot Persona Attribute Extraction in Dialogues},
  author = {Zhu, Luyao and Li, Wei and Mao, Rui and Pandelea, Vlad and Cambria, Erik},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year = {2023}
}
```



