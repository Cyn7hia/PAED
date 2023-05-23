'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''
import json
import random

import torch
from model.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from model.types_ import *
from model.layers.encoder import EncoderRNN
from model.layers.decoder import DecoderRNN
from model.layers.feedforward import FeedForward
from model.layers.loss import masked_cross_entropy, bag_of_words_loss
from model.utils.vocab import PAD_ID, EOS_ID


class MetaVAE(BaseVAE):


    def __init__(self,
                 config,
                 **kwargs) -> None:
        super(MetaVAE, self).__init__()
        self.config = config.model_params
        self.kld_weight = config.exp_params.kld_weight
        self.bow_weight = config.exp_params.bow_weight
        if len(config.trainer_params.gpus) > 0:
            self.cpu = False
        else:
            self.cpu = True
        # self.latent_dim = latent_dim

        # modules = []
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]
        #
        # # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size= 3, stride= 2, padding  = 1),
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim

        # self.encoder = nn.Sequential(*modules)
        self.embed_class = nn.Linear(self.config.num_classes, self.config.encoder_hidden_size)
        self.encoder = EncoderRNN(self.config.vocab_size,
                                         self.config.embedding_size,
                                         self.config.encoder_hidden_size,
                                         num_layers=self.config.num_layers,
                                         bidirectional=self.config.bidirectional,
                                         dropout=self.config.dropout,
                                        cpu=self.cpu
                                )
        if self.config.bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        self.fc_mu = nn.Linear(self.direction * self.config.encoder_hidden_size, self.config.latent_dim)
        self.fc_var = nn.Linear(self.direction * self.config.encoder_hidden_size, self.config.latent_dim)


        # Build Decoder
        # modules = []

        self.decoder_input = nn.Linear(self.config.latent_dim, self.direction * self.config.decoder_hidden_size)

        # hidden_dims.reverse()
        #
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )



        # self.decoder = nn.Sequential(*modules)
        self.decoder = DecoderRNN(self.config.vocab_size,
                                         self.config.embedding_size,
                                         self.config.decoder_hidden_size,
                                         num_layers=self.config.num_layers,
                                         dropout=self.config.dropout,
                                         word_drop=self.config.word_drop,
                                         max_unroll=self.config.max_unroll,
                                         sample=self.config.sample,
                                         temperature=self.config.temperature,
                                         beam_size=self.config.beam_size,
                                  cpu=self.cpu)

        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels=3,
        #                               kernel_size=3, padding=1),
        #                     nn.Tanh())
        if self.config.bow:
            self.bow_h = FeedForward(self.config.latent_dim,
                                            self.config.decoder_hidden_size,
                                            num_layers=1,
                                            hidden_size=self.config.decoder_hidden_size,
                                            activation=self.config.activation)
            self.bow_predict = nn.Linear(self.config.decoder_hidden_size, self.config.vocab_size)

        self.recons_loss = masked_cross_entropy
        # self.bow_loss = compute_bow_loss

    def encode(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        inp, input_length, embedded_class = inputs
        _, hidden = self.encoder(inp, input_length, hidden=embedded_class)
        hidden = hidden.view(self.config.num_layers, self.direction, hidden.size(1), hidden.size(-1))[-1]
        hidden = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)
        # result = torch.flatten(hidden, start_dim=1)


        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        return [mu, log_var]

    def decode(self, inputs: List[Tensor]) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        target_sent, z = inputs
        z_sent = self.decoder_input(z)
        self.z_sent = z
        z_sent = z_sent.contiguous().view(-1, self.direction, self.config.decoder_hidden_size).permute(1, 0, 2)
        # result = result.view(-1, 512, 2, 2)
        result = self.decoder(target_sent, init_h=z_sent)
        # result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        input_length = kwargs['input_length']
        y = F.one_hot(kwargs['relations'].squeeze(1), num_classes=self.config.num_classes).float()
        embedded_class = self.embed_class(y)
        mu, log_var = self.encode([input, input_length, embedded_class.unsqueeze(0).repeat(4, 1, 1)])
        z = self.reparameterize(mu, log_var)

        return [self.decode([input, z]), input, mu, log_var]

    def compute_bow_loss(self, target_sent) -> Tensor:
        target_bow = F.one_hot(target_sent, self.config.vocab_size).sum(1)
        # remove count of PAD_ID and EOS_ID
        target_bow[:, PAD_ID] = 0
        target_bow[:, EOS_ID] = 0
        bow_logits = self.bow_predict(self.bow_h(self.z_sent))
        bow_loss = bag_of_words_loss(bow_logits, target_bow)
        return bow_loss

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        inp = args[1]
        mu = args[2]
        log_var = args[3]
        input_length = kwargs['input_length']

        # kld_weight = kwargs['M_N'] if 'M_N' in kwargs else 10.  # Account for the minibatch samples from the dataset
        kld_weight = self.kld_weight
        # recons_loss = F.mse_loss(recons, input)
        recons_loss, n_words = self.recons_loss(recons, inp, input_length, cpu=self.cpu)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss/n_words + kld_weight * kld_loss
        if self.config.bow:
            bow_loss = self.compute_bow_loss(inp)
            loss += self.bow_weight * bow_loss/n_words
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach().item()/n_words.item(), 'KLD': kld_loss.detach().item()
                , 'BoW_Loss': bow_loss.detach().item()/n_words.item()}
        else:
            return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach().item() / n_words.item(),
                    'KLD': kld_loss.detach().item()}



    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def vae_sampling(model):

    inp=[0]
    with open("", 'r') as f:
        data = json.load(f)
        inp = gen_input(data)

    keys = inp.keys()
    params = []
    model.eval()
    with torch.no_grad():
        for key in keys:
            sample_i = random.choice(inp[key])['utt']
            _, _, mu, logvar = model(sample_i)
            params.append([mu, logvar])

    return [0]