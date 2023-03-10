import argparse
from model_utils import *
from transformers import StoppingCriteriaList
from utils import StopWordsCriteria


def generate(task_info, device, model, tokenizer):
    contexts = task_info["prompt_seqs"]
    inputs = tokenizer(contexts, return_tensors="pt").to(device)
    print(f"start_ids: length ({inputs.input_ids.shape[0]}) ids: {inputs.input_ids}")
    input_length = inputs.input_ids.shape[1]

    if task_info["temperature"] == 0:
        outputs = model.generate(
            **inputs, do_sample=True, top_p=task_info['top_p'],
            temperature=1.0, top_k=1,
            max_new_tokens=task_info["output_len"],
            return_dict_in_generate=True,
            output_scores=True,  # return logit score
            output_hidden_states=False,  # return embeddings
        )
    else:
        outputs = model.generate(
            **inputs, do_sample=True, top_p=task_info['top_p'],
            temperature=task_info["temperature"],
            max_new_tokens=task_info["output_len"],
            return_dict_in_generate=True,
            output_scores=True,  # return logit score
            output_hidden_states=False,  # return embeddings
            stopping_criteria=StoppingCriteriaList([StopWordsCriteria(task_info["stop"], tokenizer)]) if task_info.get("stop") else None,
        )
    print(f"[INFO] raw output: {outputs}")
    token = outputs.sequences[0, input_length:]  # exclude context input from the output
    print(f"[INFO] raw token: {token}")
    output = tokenizer.decode(token)
    print(f"[INFO] \n[Context]\n{contexts}\n\n[Output]\n{output}\n")


def test_model(args):
    print(f"<test_model> initialization start")
    device = torch.device(args.get('device', 'cuda'))
    assert args['model_path'] != ''
    model, tokenizer = get_local_huggingface_tokenizer_model(args['hf_model_name'], args['model_path'], args.get('dtype'))
    model = model.to(device)
    tokenizer = tokenizer
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    task_info = {
        "seed": 0,
        "prompt_seqs": None,
        "output_len": 16,
        "beam_width": 1,
        "top_k": 50,
        "top_p": 1,
        "beam_search_diversity_rate": 0,
        "temperature": 0.8,
        "len_penalty": 0,
        "repetition_penalty": 1.0,
        "stop": args.get("stop", []),
        "logprobs": 5,
    }
    print(f"<test_model> initialization done")

    if args["interactive"]:
        while True:
            prompt_data = input("Please enter the prompt input:\n")
            task_info["prompt_seqs"] = prompt_data
            generate(task_info, device, model, tokenizer)
    else:
        generate(task_info, device, model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str, default='facebook/opt-350m',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--model_path', type=str, default='~/models/opt-350m',
                        help='hugging face model path (used to load config).')
    args = parser.parse_args()
    test_model(args={
        "hf_model_name": args.hf_model_name,
        "model_path": args.model_path,
        "interactive": True,
        # "stop": ["pioneering"],
        # "device": "cpu",
        # "dtype": torch.float32,
    })
