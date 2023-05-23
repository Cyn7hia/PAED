'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, BartForConditionalGeneration)
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right, BartModel
from transformers.utils import (
    add_end_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from typing import List, Any, Dict, Optional, Union, Tuple

from config import ModelArguments
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "BartConfig"

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartConGen(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs  # outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class RelationExt(nn.Module):
    def __init__(self, train_args) -> None:
        super(RelationExt, self).__init__()
        self.model_name = "facebook/bart-base"
        model_args = self.get_args()

        self.build_model(model_args, self.config)


        # self.label_smoother = LabelSmoother(epsilon=train_args.label_smoothing_factor,
        #                                         ignore_index=self.config.pad_token_id)
        # Label smoothing
        if train_args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=train_args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def get_args(self):
        model_args = ModelArguments(
            model_name_or_path=self.model_name
        )
        self.config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            output_hidden_states=True,
        )
        return model_args

    def build_model(self, model_args, config) -> None:

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # self.model = BartConGen(config)

        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,  # 1
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        # if torch.cuda.is_available() and train_args.device.type == 'cuda':
        #     self.model.cuda()
        #     self.classification_head.cuda()

    def forward_(self, inputs, model_type='extraction'):

        outputs = self.model(**inputs)  # come from trainer; todo: check if the model is BartModel or BartForConditionalGeneration; ans: BartForConditionalGeneration
        if model_type == 'extraction':
            return outputs.logits
        # batch_size = outputs[0].size(0)
        batch_size = outputs.logits.size(0)
        # hidden_states = outputs[0][int(batch_size/3):]  # last hidden state
        hidden_states = outputs.decoder_hidden_states[-1][int(batch_size/3):]  # last hidden state

        eos_mask = inputs['input_ids'].eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :]
        cls_logits = self.classification_head(sentence_representation)

        return outputs.logits[:int(batch_size/3)], cls_logits

    def forward(self, inputs, model_type='both'):


        if model_type == 'extraction':
            outputs = self.model(**inputs)  # come from trainer; todo: check if the model is BartModel or BartForConditionalGeneration; ans: BartForConditionalGeneration
            # return outputs.logits
            return outputs
        # batch_size = outputs[0].size(0)
        ext_inps, cnt_inps = inputs
        ext_outputs = self.model(**ext_inps)  # come from trainer; todo: check if the model is BartModel or BartForConditionalGeneration; ans: BartForConditionalGeneration
        cnt_outputs = self.model(**cnt_inps)
        # batch_size = outputs.logits.size(0)
        # hidden_states = outputs[0][int(batch_size/3):]  # last hidden state
        hidden_states = cnt_outputs.decoder_hidden_states[-1]  # last hidden state

        eos_mask = cnt_inps['input_ids'].eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :]
        cls_logits = self.classification_head(sentence_representation)

        # return ext_outputs.logits, cls_logits
        return ext_outputs, cls_logits

    def compute_ce_loss(self, model, inputs, outputs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return loss

    def compute_cnt_loss(self, logits, temperature=1, k=1, kl_type='sum'):
        batch_size, dim = logits.size()
        p_logits = logits[: int(batch_size/(k+1))].unsqueeze(1).repeat(1,k,1).contiguous().view(-1, dim)
        pos_logits = F.log_softmax(p_logits, dim=-1)
        neg_logits = F.log_softmax(logits[int(batch_size/(k+1)):], dim=-1)
        pos_softs = F.softmax(p_logits, dim=-1)
        neg_softs = F.softmax(logits[int(batch_size/(k+1)):], dim=-1)

        if kl_type == "sum":
            kl_loss = -(self.kl_loss(pos_logits/temperature, neg_softs) + self.kl_loss(neg_logits/temperature, pos_softs))
        elif kl_type == "max":
            kl_loss = -max(self.kl_loss(pos_logits, neg_softs), self.kl_loss(neg_logits, pos_softs))
        else:
            kl_loss = -self.kl_loss(neg_logits/temperature, pos_softs)

        kl_loss = kl_loss / k

        return kl_loss

    def run(
        self,
        inputs: Any,  # Dict[Tensor],
        tokenizer: Any,
        do_sample=False,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        save_scores: bool = False,
        max_length: int = 128,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        if type(inputs) is list:
            tok = self.tokenizer
            eos, bos = tok.eos_token_id, tok.bos_token_id

            if prompt is not None:
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            if prompt_ids is not None:
                prompt_ids = [eos, bos] + prompt_ids
                decoder_input_ids = torch.tensor([prompt_ids])
            if multi_prompt_ids is not None:
                assert len(inputs) == len(multi_prompt_ids)
                multi_prompt_ids = [[eos, bos] + lst for lst in multi_prompt_ids]
                decoder_input_ids = torch.tensor(multi_prompt_ids)
            if decoder_input_ids is not None:
                kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))

            inputs = self.tokenize(inputs, max_length=max_length)

        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return,
            return_dict_in_generate=True,
            output_scores=save_scores,
            max_length=max_length,
            **kwargs,
        )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        return self.decode(outputs.sequences, tokenizer)

    def decode(self, outputs, tokenizer) -> List[str]:
        tok = tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.bos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts

    def tokenize(self, texts: List[str], max_length: int, **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

