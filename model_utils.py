import torch
from transformers import AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM
import logging

logger = logging.getLogger(__name__)

def get_local_huggingface_tokenizer_model(model_name, model_path=None, dtype=None):
    if model_name.startswith('Salesforce/codegen'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_path is not None:
            print(f"<get_local_huggingface_tokenizer_model> Load from path: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    elif model_name == 'facebook/opt-350m':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=(dtype if dtype else torch.float16))
    elif model_name == 'google/flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        if model_path is not None:
            print(f"<get_local_huggingface_tokenizer_model> Load from path: {model_path}")
            model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        else:
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.bfloat16)
    elif model_name == 'facebook/opt-iml-30b':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-30b", use_fast=False)
        if model_path is not None:
            print(f"<get_local_huggingface_tokenizer_model> Load from path: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-iml-30b", torch_dtype=torch.float16)
    elif model_name == "chip_20B_instruct_alpha":
        assert model_path is not None
        print(f"<get_local_huggingface_tokenizer_model> Load from path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, load_in_8bit=False)
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
    elif model_name == 'stabilityai/stablelm-base-alpha-3b':
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b", torch_dtype=torch.float16)
    elif model_name == 'stabilityai/stablelm-base-alpha-7b':
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-7b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-7b", torch_dtype=torch.float16)
    elif model_name == 'EleutherAI/gpt-neox-20b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16)
    elif model_name == 'Together/gpt-neoxT-20b':
        if model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            assert False
    elif model_path is not None and model_path != "":
        logger.warning("model_path is not None, but model_name is not given. Load from model_path only")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        assert False, "Model not supported yet."

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    return model, tokenizer


def get_local_huggingface_tokenizer_model_llm_int8(model_name, model_path=None, dtype=None):
    
    if model_path is None:
        model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', load_in_8bit=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    return model, tokenizer


def get_dist_accelerate_tokenizer_model(model_name, model_path):
    from accelerate import init_empty_weights,load_checkpoint_and_dispatch
    if model_name == "facebook/galactica-120b":
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            model = load_checkpoint_and_dispatch(
                model, model_path, device_map="auto", no_split_module_classes=["OPTDecoderLayer"]
            )
            tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-120b")
    elif model_name == "facebook/opt-iml-175b-max":
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            # state_dict = torch.load(model_path+'/opt-iml-max.pt')
            # model.load_state_dict(state_dict)
            model = load_checkpoint_and_dispatch(
                model, model_path+'/opt-iml-max.pt', device_map="auto", no_split_module_classes=["OPTDecoderLayer"]
            )
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-30b", use_fast=False)
    elif model_name == "facebook/opt-iml-175b-regular":
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            model = load_checkpoint_and_dispatch(
                model, model_path+'/opt-iml-regular.pt', device_map="auto", no_split_module_classes=["OPTDecoderLayer"]
            )
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-30b", use_fast=False)
    else:
        assert False, f"Not legal name {model_name}"
    print(f"<get_dist_accelerate_tokenizer_model>: {model_name} hf_device_map")
    print(model.hf_device_map)
    return model, tokenizer


def get_dist_alpa_tokenizer_model(model_name, model_path):
    from llm_serving.model.wrapper import get_model
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

