# PAED: Zero-Shot Persona Attribute Extraction in Dialogues

## Overview
This is a project for our paper ([PAED: Zero-Shot Persona Attribute Extraction in Dialogues](/paper/PAED.pdf)) accepted in ACL'23 ([The 61st Annual Meeting of the Association for Computational Linguistics](https://2023.aclweb.org)).
>We develop a PAED dataset, PersonaExt, with 1,896 re-annotated triplets and 6,357 corrected utterance- triplet pairs. 
>We present a generation-based framework for zero-shot PAED. A novel HNS strat- egy, Meta-VAE sampler with CSC, is presented to enhance the performance of our model.
>Our model achieves better results than strong baselines in zero-shot PAED and negative sampling.

## Uasge
To run this code, please use the following command (take split `unseen_10_seed_0` as an example). 
>10 is the number of unseen labels; 0 is the random seed used for randomly selecting the 10 unseen labels.

```
python trainer_finetune.py
```



