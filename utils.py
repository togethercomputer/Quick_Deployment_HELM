import torch
from transformers import AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AutoConfig, AutoTokenizer
from llm_serving.model.wrapper import get_model



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


def get_local_huggingface_tokenizer_model(model_name):
    if model_name.startswith('Salesforce/codegen'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name == 'facebook/opt-350m':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
    elif model_name == 'google/flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.bfloat16)
    elif model_name == 't5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b', model_max_length=512)
        # tokenizer.model_max_length=512
        model = T5ForConditionalGeneration.from_pretrained('t5-11b', torch_dtype=torch.bfloat16)
        model.config.eos_token_id = None
    elif model_name == 'google/ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/ul2')
        model = T5ForConditionalGeneration.from_pretrained("google/ul2", torch_dtype=torch.bfloat16)
    elif model_name == 'EleutherAI/gpt-j-6b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
    elif model_name == 'togethercomputer/GPT-JT-6B-v1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1", torch_dtype=torch.float16)
    elif model_name == 'EleutherAI/gpt-neox-20b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16)
    else:
        assert False, "Model not supported yet."

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    return model, tokenizer


def get_dist_alpa_tokenizer_model(model_name, model_path):
    if model_name == 'opt-2.7b':
        # The 30B version works for all OPT models.
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/opt-2.7b", path=model_path)
    elif model_name == 'opt-175b':
        # The 30B version works for all OPT models.
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/opt-175b", path=model_path)
    elif model_name == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom')
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/bloom", path=model_path)
    elif model_name == 'bloomz':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloomz')
        tokenizer.add_bos_token = False
        # llm_serving does not recoginze bloomz, since the model parameter is from bloomz,
        # this should be fine
        model = get_model(model_name="alpa/bloom", path=model_path)
    else:
        assert False, f"Not legal name {model_name}"

    return model, tokenizer





def convert_hf_score_to_logprobs(scores, k, tokenizer):
    logprobs = []
    for current_step_score in scores:
        print(current_step_score.shape)
        value, indices = torch.topk(torch.log_softmax(torch.squeeze(current_step_score), dim=0), k)
        current_logprob = list(zip(tokenizer.convert_ids_to_tokens(indices.tolist()), value.tolist()))
        logprobs.append(current_logprob)
    return logprobs
