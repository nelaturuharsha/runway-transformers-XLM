#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
from transformers import XLMConfig
from transformers import XLMWithLMHeadModel, XLMTokenizer
from runway.data_types import *
import runway

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


MODEL_CLASSES = {
    'xlm': (XLMWithLMHeadModel, XLMTokenizer)
}


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, xlm_lang, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            
            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

           
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


@runway.setup(options={"model_lang" : category(choices=['French', 'German'], default='French')})
def setup(opts):
    model_lang = opts["model_lang"] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    if model_lang == 'French':
        model_type = 'xlm-clm-enfr-1024'
    else:
        model_type = 'xlm-clm-ende-1024'

    seed = 42
    set_seed(seed, n_gpu)

    model_name = 'xlm'
    print(model_type)
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    print(model_class, tokenizer_class)
    tokenizer = tokenizer_class.from_pretrained(model_type)
    model = model_class.from_pretrained(model_type)
    model.to(device)
    model.eval()

    
    
    return {"model_type" : model_type,
            "model" : model,
            "tokenizer" : tokenizer, 
            "device" : device}

command_inputs = {
    "input_prompt" : text, 
    "length" : number(min=20, default=20, step=1, description="Output Text Length"),
    "temperature" : number(default=1.0, step=0.1, max=4, description="Temperature of output distribution")
}

command_outputs = {"generated_text" : text}

@runway.command("generated_text", inputs=command_inputs, outputs=command_outputs, description="Generate text conditioned on prompt")
def generate_text(model_opts, inputs):
    
    model_type = model_opts["model_type"]
    model = model_opts["model"]
    tokenizer = model_opts["tokenizer"]
    device = model_opts["device"]

    length = inputs["length"]
    num_samples = 1
    temperature = inputs["temperature"]
    repetition_penalty = 1.0
    top_k = 1
    top_p = 0.9
    no_cuda = torch.cuda.is_available()
    stop_token = 'None'
    
    if length < 0 and model.config.max_position_embeddings > 0:
        length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < length:
        length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop

    if model_type == "xlm-clm-enfr-1024":
        xlm_lang = 'fr'
    else:
        xlm_lang = 'de'


    while True:
        
        language = xlm_lang
        xlm_lang = tokenizer.lang2id[language]
        

        raw_text = inputs["input_prompt"]
        
        # Models with memory likes to have a long prompt for short inputs.
        
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        
        out = sample_sequence(
            model=model,
            context=context_tokens,
            num_samples=num_samples,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
            xlm_lang=xlm_lang
        )
        out = out[:, len(context_tokens):].tolist()
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(stop_token) if stop_token else None]

        if raw_text:
            break
    return text


if __name__ == '__main__':
    runway.run()
