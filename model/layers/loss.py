'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''
import torch
from torch.nn import functional as F
from model.utils.mask import sequence_mask
import numpy as np
from collections import Counter
from model.utils.vocab import PAD_ID, EOS_ID


# https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def masked_cross_entropy(logits, target, length, cpu=False, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len, cpu=cpu)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    # loss = losses.sum() / length.float().sum()

    if per_example:
        # loss: [batch_size]
        return losses.sum(1)
    else:
        loss = losses.sum()
        return loss, length.float().sum()


def to_bow(sentence, vocab_size):
    '''  Convert a sentence into a bag of words representation
    Args
        - sentence: a list of token ids
        - vocab_size: V
    Returns
        - bow: a integer vector of size V
    '''
    bow = Counter(sentence)
    # Remove EOS tokens
    bow[PAD_ID] = 0
    bow[EOS_ID] = 0

    x = np.zeros(vocab_size, dtype=np.int64)
    x[list(bow.keys())] = list(bow.values())

    return x


def bag_of_words_loss(bow_logits, target_bow, weight=None):
    ''' Calculate bag of words representation loss
    Args
        - bow_logits: [num_sentences, vocab_size]
        - target_bow: [num_sentences]
    '''
    log_probs = F.log_softmax(bow_logits, dim=1)
    target_distribution = target_bow / (target_bow.sum(1).view(-1, 1) + 1e-23) + 1e-23
    entropy = -(torch.log(target_distribution) * target_bow).sum()
    loss = -(log_probs * target_bow).sum() - entropy

    return loss