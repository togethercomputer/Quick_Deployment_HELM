import torch
from transformers import AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AutoConfig, AutoTokenizer


def get_int(input_: str, default=0) -> int:
    try:
        my_num = int(input_)
        return my_num
    except ValueError:
        print(f'Invalid int {input_} set to default: {default}')
        return default


def get_float(input_: str, default=0.0) -> float:
    try:
        my_num = float(input_)
        return my_num
    except ValueError:
        print(f'Invalid float {input_} set to default: {default}')
        return default


def post_processing_text(output_text, stop_tokens):
    print(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)

    print(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    print(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    print(f"<post_processing_text>2 end_pos: {end_pos}.")
    print(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


def get_huggingface_tokenizer_model(model_name):
    if model_name == 'facebook/opt-350m':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
    elif model_name == 'flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.bfloat16)
    elif model_name == 't5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b', model_max_length=512)
        # tokenizer.model_max_length=512
        model = T5ForConditionalGeneration.from_pretrained('t5-11b', torch_dtype=torch.bfloat16)
        model.config.eos_token_id = None
    elif model_name == 't0pp':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/T0pp')
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", torch_dtype=torch.bfloat16)
    elif model_name == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/ul2')
        model = T5ForConditionalGeneration.from_pretrained("google/ul2", torch_dtype=torch.bfloat16)
    elif model_name == 'gpt-j-6b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
    elif model_name == 'Together-gpt-JT-6B-v1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float16)
    elif model_name == 'gpt-neox-20b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16)
    else:
        assert False, "Model not supported yet."

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    return model, tokenizer


def convert_hf_score_to_logprobs(scores, k, tokenizer):
    logprobs = []
    for current_step_score in scores:
        print(current_step_score.shape)
        # assume batch_size = 1.
        value, indices = torch.topk(torch.log_softmax(torch.squeeze(current_step_score), dim=0), k)
        # print(f"{value}, {indices}")
        # print(f"{tokenizer.convert_ids_to_tokens(indices.tolist(), skip_special_tokens=True)}")
        current_logprob = list(zip(tokenizer.convert_ids_to_tokens(indices.tolist()), value.tolist()))
        # print(f"{current_logprob}")
        logprobs.append(current_logprob)
    return logprobs
