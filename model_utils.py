import torch
from transformers import AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)

def get_local_huggingface_tokenizer_model(
        model_name, 
        model_path=None, 
        dtype=None, 
        auth_token=None, 
        max_memory=None, 
        trust_remote_code=False, 
        device=None,
        lora_adapters="",
        quantize=False,
): 
    if max_memory != {}:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        # load empty weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        model.tie_weights()
            
        #create a device_map from max_memory
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["GPTNeoXLayer", "DecoderLayer"],
            dtype=dtype,
        )
    else:
        device_map = None

    if quantize:
        load_in_4bit = True
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
    else:
        load_in_4bit = False
        quantization_config = None

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
    elif model_name == 'EleutherAI/gpt-neox-20b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16)

    # DataBricks models
    elif model_name == 'databricks/dolly-v2-3b':
        tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
        model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
    elif model_name == 'databricks/dolly-v2-7b':
        tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b")
        model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b")
    elif model_name == 'databricks/dolly-v2-12b':
        tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
        model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")

    # StabilityAI Models
    elif model_name == 'stabilityai/stablelm-base-alpha-3b':
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-3b")
    elif model_name == 'stabilityai/stablelm-base-alpha-7b':
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-7b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-7b")

    # Together Computer Models
    elif model_name == 'togethercomputer/Pythia-Chat-Base-7B-v0.16':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Pythia-Chat-Base-7B-v0.16")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/Pythia-Chat-Base-7B-v0.16")
    elif model_name == 'togethercomputer/GPT-JT-6B-v1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-6B-v1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-6B-v1")
    elif model_name == 'togethercomputer/GPT-JT-X-6B-v1.1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-X-6B-v1.1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-X-6B-v1.1")
    elif model_name == 'togethercomputer/GPT-JT-Moderation-6B':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-JT-Moderation-6B")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-JT-Moderation-6B")
    elif model_name == 'togethercomputer/GPT-NeoXT-Chat-Base-20B':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", torch_dtype=torch.float16)

    elif model_name == 'togethercomputer/RedPajama-INCITE-Base-3B-v1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
    elif model_name == 'togethercomputer/RedPajama-INCITE-Chat-3B-v1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
    elif model_name == 'togethercomputer/RedPajama-INCITE-Instruct-3B-v1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
    elif model_name == 'togethercomputer/RedPajama-INCITE-Base-7B-v0.1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1")
    elif model_name == 'togethercomputer/RedPajama-INCITE-Chat-7B-v0.1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-7B-v0.1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-7B-v0.1")
    elif model_name == 'togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1':
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1")

    elif model_name == 'openlm-research/open_llama_7b_preview_200bt':
        tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b_preview_200bt")
        model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b_preview_200bt")

    elif model_name == 'Together/gpt-neoxT-20b':
        if model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            assert False
    elif model_path is not None and model_path != "":
        logger.warning("model_path is not None, but model_name is not given. Load from model_path only")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_auth_token=auth_token,
            trust_remote_code=trust_remote_code
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
            device_map=device_map,
            trust_remote_code=trust_remote_code
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=auth_token,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=auth_token,
            device_map=device_map if lora_adapters != "" else None,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            load_in_4bit=load_in_4bit,
            quantization_config=quantization_config,
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    if lora_adapters != "":
        model = PeftModel.from_pretrained(model, lora_adapters)

    if max_memory == {}:
        model = model.to(device)

    return model, tokenizer


def get_local_huggingface_tokenizer_model_llm_int8(model_name, model_path=None, dtype=None, auth_token=None):
    if model_path is None:
        model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', load_in_8bit=True, use_auth_token=auth_token)

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

