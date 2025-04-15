# Modified from LLaVA: https://github.com/haotian-liu/LLaVA.git
#
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Model, Phi3Config, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from model.tf.modeling_outputs import CausalLMOutputWithPastAndLabel
from transformers.generation.utils import GenerateOutput

from ..llava_arch_ import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaPhiConfig(Phi3Config):
    model_type = "llava_phi"


class LlavaPhiModel(LlavaMetaModel, Phi3Model):
    config_class = LlavaPhiConfig

    def __init__(self, config: Phi3Config):
        super(LlavaPhiModel, self).__init__(config)


class LlavaPhiForCausalLM(Phi3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        dense_indices: Optional[List[int]] = None,
        temporal_tokens: Optional[List[torch.Tensor]] = None, 
        extracted_images: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndLabel]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # True
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # True

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                temporal_tokens=temporal_tokens, 
                # image_sizes,
                dense_indices=dense_indices, 
                extracted_images=extracted_images
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # if self.training:
        #     output_hidden_states = outputs.hidden_states
        # else:
        #     output_hidden_states = hidden_states
            
        output_hidden_states = outputs.hidden_states

        return CausalLMOutputWithPastAndLabel(
            loss=loss,
            labels=labels,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        dense_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                dense_indices=dense_indices,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_phi", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)
