import torch
from transformers import AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map
from huggingface_hub import snapshot_download, login
import logging

logger = logging.getLogger(__name__)

def get_local_huggingface_tokenizer_model(
        model_name, 
        dtype=None, 
        auth_token=None, 
        max_memory=None, 
        trust_remote_code=False, 
        device=None,
): 
    login(token=auth_token)
    # pre-download the model with 8 parallel workers
    weights_path = snapshot_download(repo_id=model_name)

    if model_name in ['google/flan-t5-xxl', 'togethercomputer/instructcodet5p-16b', 'google/flan-t5-xl', 'lmsys/fastchat-t5-3b-v1.0']:
        tokenizer = T5Tokenizer.from_pretrained(weights_path, skip_special_tokens=True)
        model = T5ForConditionalGeneration.from_pretrained(weights_path, torch_dtype=torch.bfloat16)
        model = model.to(device)
    elif model_name == 't5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b', model_max_length=512)
        # tokenizer.model_max_length=512
        model = T5ForConditionalGeneration.from_pretrained('t5-11b', torch_dtype=torch.bfloat16)
        model.config.eos_token_id = None
        model = model.to(device)
    else:
        if max_memory == {}:
            max_memory = None

        config = AutoConfig.from_pretrained(weights_path, trust_remote_code=trust_remote_code)
        # load empty weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        model.tie_weights()
    
        #create a device_map from max_memory
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["GPTNeoXLayer", "DecoderLayer", "LlamaDecoderLayer", "MPTBlock", "CodeGenBlock"],
            dtype=dtype,
        )
                
        tokenizer = AutoTokenizer.from_pretrained(
            weights_path,
            use_auth_token=auth_token,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            skip_special_tokens=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            weights_path,
            use_auth_token=auth_token,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    return model, tokenizer


def get_local_huggingface_tokenizer_model_llm_int8(model_name, dtype=None, auth_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token, skip_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True, use_auth_token=auth_token)

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
    elif model_name == "huggyllama/llama-65b":
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(
            model, model_path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == "together/falcon-40b" or model_name == "together/falcon-40b-instruct":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(
            model, model_path, device_map="auto", no_split_module_classes=["FalconDecoderLayer"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
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

