'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from model.utils.convert import to_var
from model.utils.vocab import PAD_ID


class BaseRNNEncoder(nn.Module):
    def __init__(self):
        """Base RNN Encoder Class"""
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, 'rnn'):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError('no rnn selected')

    def init_h(self, batch_size=None, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            return (to_var(torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size), on_cpu=self.cpu),
                    to_var(torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size), on_cpu=self.cpu))
        else:
            return to_var(torch.zeros(self.num_layers*self.num_directions,
                                        batch_size,
                                        self.hidden_size), on_cpu=self.cpu)

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTM)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self):
        raise NotImplementedError


class EncoderRNN(BaseRNNEncoder):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, rnn=nn.GRU, num_layers=1, bidirectional=False,
                 dropout=0.0, bias=True, batch_first=True, cpu=False):
        """Sentence-level Encoder"""
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.cpu = cpu

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_ID)

        self.rnn = rnn(input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, inputs, input_length, hidden=None):
        """
        Args:
            inputs (Variable, LongTensor): [num_setences, max_seq_len]
            input_length (Variable, LongTensor): [num_sentences]
        Return:
            outputs (Variable): [max_source_length, batch_size, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len = inputs.size()

        # Sort in decreasing order of length for pack_padded_sequence()
        input_length_sorted, indices = input_length.sort(descending=True)

        # input_length_sorted = input_length_sorted#.data.tolist()

        # [num_sentences, max_source_length]
        inputs_sorted = inputs.index_select(0, indices)

        # [num_sentences, max_source_length, embedding_dim]
        embedded = self.embedding(inputs_sorted)

        # batch_first=True
        rnn_input = pack_padded_sequence(embedded, input_length_sorted.cpu(),
                                            batch_first=self.batch_first)

        hidden = self.init_h(batch_size, hidden=hidden)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)
        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=self.batch_first)

        # Reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                        hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        return outputs, hidden