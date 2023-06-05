import os
import sys
import math
import json
import torch
import timeit
import random
import logging
import argparse
import numpy as np
from faiss_retrieval import *
from utils import *
from model_utils import *
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, StoppingCriteriaList
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions

logger = logging.getLogger(__name__)

logger.setLevel(int(os.environ.get('LOG_LEVEL', logging.DEBUG)))

def translate_chatml_to_openchat(prompt):
    prompt = prompt.replace('<|im_start|>system\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>user\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>assistant\n', '<bot>: ')
    prompt = prompt.replace('<|im_start|>user', '<human>:')
    prompt = prompt.replace('<|im_start|>assistant', '<bot>:')
    prompt = prompt.replace('\n<|im_end|>', '')
    prompt = prompt.replace('<|im_end|>', '')
    prompt = prompt.rstrip()
    # print(prompt)
    return prompt

class HuggingFaceLocalNLPModelInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        logging.debug(f"Model name: {model_name}")
        logging.debug("\n=============== Arguments ===============")
        logging.debug(args.keys())
        logging.debug(args)
        logging.debug("=========================================\n")
        self.task_info = {
            "seed": 0,
            "prompt_seqs": None,
            "output_len": 16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "stop": [],
            "logprobs": 0,
        }
        self.device = args['device']
        self.hf_model_name = args['hf_model_name']
        self.max_batch_size = args['max_batch_size']
        self.deny_list = args['deny_list']
        
        auth_token = os.environ.get("AUTH_TOKEN")
        
        if args.get('dtype') == 'llm.int8':
            model, tokenizer = get_local_huggingface_tokenizer_model_llm_int8(args['hf_model_name'], args['model_path'], None, auth_token=auth_token)
            self.model = model # int8 cannot do .to(device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self.tokenizer = tokenizer
        else:
            if args['model_path'] != '':
                model, tokenizer = get_local_huggingface_tokenizer_model(args['hf_model_name'], args['model_path'], args.get('dtype'), auth_token=auth_token)
            else:
                model, tokenizer = get_local_huggingface_tokenizer_model(args['hf_model_name'], None, args.get('dtype'), auth_token=auth_token)
            self.model = model.to(self.device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self.tokenizer = tokenizer
        self.plugin = args.get('plugin')
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        logging.debug(f"<HuggingFaceLocalNLPModelInference.__init__> initialization done")

    def dispatch_request(self, args, env) -> Dict:
        plugin_state = {}
        if self.plugin:
            self.plugin.request(args, env, plugin_state)
        logging.debug(f"<HuggingFaceLocalNLPModelInference.dispatch_request> starts")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["seed"] = get_int(args.get("seed", 0), default=0)
        if isinstance(str(args['prompt']), str):
            self.task_info["prompt_seqs"] = [str(args['prompt'])]
        elif isinstance(str(args['prompt']), list):
            self.task_info["prompt_seqs"] = args['prompt']
        else:
            logging.debug("wrong prompt format, it can only be str or list of str")
            return
        self.task_info["output_len"] = get_int(args.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(args.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(args.get("top_k", 50), default=50)
        if self.hf_model_name == "google/flan-t5-xxl":
            self.task_info["top_p"] = get_float(args.get("top_p", 1.0), default=1.0)
        else:
            self.task_info["top_p"] = get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(args.get("beam_search_diversity_rate", 0.0),
                                                                 default=0.0)
        self.task_info["temperature"] = get_float(args.get("temperature", 0.8), default=0.8)
        self.task_info["len_penalty"] = get_float(args.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(args.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["logprobs"] = get_int(args.get("logprobs", 0), default=0)

        if args.get("stream_tokens"):
            self.task_info["stream_tokens"] = lambda token: self.stream_tokens(token, env)

        if len(self.task_info["prompt_seqs"][0]) == 0 or self.task_info["output_len"] == 0:
            inference_result = []
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {
                    "text": '',
                    "index": beam_id,
                    "finish_reason": "length"
                }
                item['choices'].append(choice)
            inference_result.append(item)
            #  So far coordinator does not support batch.
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inference_result[0]['choices'],
                "raw_compute_time": 0.0
            }
            logging.debug(f"<HuggingFaceLocalNLPModelInference.dispatch_request> (empty input or output) return: {result}")
            if self.plugin:
                return self.plugin.response(result, plugin_state)
            return result
        else:
            result = self._run_inference()
            torch.cuda.empty_cache()
            logging.debug(f"<HuggingFaceLocalNLPModelInference.dispatch_request> return: {result}")
            if self.plugin:
                return self.plugin.response(result, plugin_state)
            return result

    def _run_inference(self):
        logging.debug(f"<HuggingFaceLocalNLPModelInference._run_inference> start.")
        complete_contexts = self.task_info["prompt_seqs"]

        with torch.no_grad():
            logging.debug(self.task_info)
            torch.manual_seed(self.task_info['seed'])
            np.random.seed(self.task_info['seed'])
            random.seed(self.task_info['seed'])
            batch_size = min(len(complete_contexts), self.max_batch_size)
            num_iter = math.ceil(len(complete_contexts) / batch_size)
            output_buffer = []
            logprobs_buffer = []
            output_scores = self.task_info["logprobs"] > 0
            if output_scores:
                logprobs_buffer = []
            else:
                logprobs_buffer = None

            time = timeit.default_timer()
            for iter_i in range(num_iter):
                contexts = complete_contexts[iter_i * batch_size: (iter_i + 1) * batch_size]
                # Do translation
                contexts = [translate_chatml_to_openchat(context) for context in contexts]
                inputs = self.tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logging.debug(f"start_ids: length ({inputs.input_ids.shape[0]}) ids: {inputs.input_ids}")
                input_length = inputs.input_ids.shape[1]

                if self.task_info["temperature"] == 0:
                    outputs = self.model.generate(
                        **inputs, do_sample=False, 
                        max_new_tokens=self.task_info["output_len"],
                        return_dict_in_generate=True,
                        output_scores=output_scores,  # return logit score
                        output_hidden_states=True,  # return embeddings
                        stream_tokens=self.task_info.get("stream_tokens"),
                    )
                else:
                    outputs = self.model.generate(
                        **inputs, 
                        do_sample=True, 
                        top_p=self.task_info['top_p'],
                        top_k=self.task_info['top_k'],
                        repetition_penalty=self.task_info['repetition_penalty'],
                        temperature=self.task_info["temperature"],
                        max_new_tokens=self.task_info["output_len"],
                        return_dict_in_generate=True,
                        output_scores=output_scores,  # return logit score
                        output_hidden_states=True,  # return embeddings
                        stream_tokens=self.task_info.get("stream_tokens"),
                        stopping_criteria=StoppingCriteriaList([StopWordsCriteria(self.task_info["stop"], self.tokenizer)]) if self.task_info.get("stop") else None,
                    )
                if output_scores:
                    ### hard code, assume bsz==1
                    n_logprobs = self.task_info["logprobs"]
    
                    # sampled tokens
                    token_ids = outputs.sequences[0, inputs['input_ids'].size(1):].tolist()
                    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                    logprobs_dict = {
                        'tokens': tokens,
                        'token_logprobs': [],
                        'top_logprobs': [],
                    }

                    # last layer hidden states
                    hids = [outputs.hidden_states[0][-1][:, -1:]]
                    hids += [hid[-1] for hid in outputs.hidden_states[1:]]
                    hids = torch.cat(hids, dim=1)
                    # origianl logits
                    logits = self.model.get_output_embeddings()(hids)
                    logprobs = logits.log_softmax(-1)
                    values, indices = logprobs.topk(n_logprobs, dim=-1)

                    for i in range(indices.size(1)):
                        selected_token_id = token_ids[i]
                        # topk tokens
                        tokens = self.tokenizer.convert_ids_to_tokens(indices[0, i])
                        # topk scores
                        scores = values[0, i].tolist()

                        logprobs_dict['token_logprobs'].append(logprobs[0, i, selected_token_id].item())
                        logprobs_dict['top_logprobs'].append({
                            t: s for t,s in zip(tokens, scores)
                        })
                        
                    logprobs_buffer.append(logprobs_dict)
                    
                output_buffer.append(outputs)
            time_elapsed = timeit.default_timer() - time

        logging.debug(f"[INFO] HuggingFaceLocalNLPModelInference time costs: {time_elapsed} ms. ")

        if len(complete_contexts) == 1:
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                if self.hf_model_name == "google/flan-t5-xxl":
                    token = outputs.sequences[beam_id, :]
                else:
                    token = outputs.sequences[beam_id, input_length:]  # exclude context input from the output
                logging.debug(f"[INFO] raw token: {token}")
                output = self.tokenizer.decode(token)
                logging.debug(f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n")
                choice = {
                    "text": post_processing_text(output, self.task_info["stop"], self.deny_list),
                    "index": beam_id,
                    "finish_reason": "length"
                }
                if output_scores:
                    choice['logprobs'] = logprobs_buffer[0]
                item['choices'].append(choice)
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": item['choices'],
                "raw_compute_time": time_elapsed
            }
        else:
            """
            inference_result = []
            for outputs in output_buffer:
                beam_width = self.task_info["beam_width"]
                current_batch_size = outputs.sequences.shape[0] // beam_width
                for sample_id in range(current_batch_size):
                    item = {'choices': [], }
                    for beam_id in range(beam_width):
                        if self.hf_model_name == "google/flan-t5-xxl":
                            token = outputs.sequences[sample_id*beam_width+beam_id, :]
                        else:
                            # exclude context input from the output
                            token = outputs.sequences[sample_id*beam_width+beam_id, input_length:]
                        logging.debug(f"[INFO] raw token: {token}")
                        output = self.tokenizer.decode(token)
                        logging.debug(f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n")
                        choice = {
                            "text": post_processing_text(output, self.task_info["stop"], self.deny_list),
                            "index": beam_id,
                            "finish_reason": "length"
                        }
                        item['choices'].append(choice)
                    inference_result.append(item)
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inference_result,
                "raw_compute_time": time_elapsed,
                # "output_length": [outputs.sequences.shape[0] for outputs in output_buffer]
            }
            """
            item = {'choices': [], }
            for i_output, outputs in enumerate(output_buffer):
                beam_width = self.task_info["beam_width"]
                current_batch_size = outputs.sequences.shape[0] // beam_width
                for sample_id in range(current_batch_size):

                    for beam_id in range(beam_width):
                        if self.hf_model_name == "google/flan-t5-xxl":
                            token = outputs.sequences[sample_id * beam_width + beam_id, :]
                        else:
                            # exclude context input from the output
                            token = outputs.sequences[sample_id * beam_width + beam_id, input_length:]
                        logging.debug(f"[INFO] raw token: {token}")
                        output = self.tokenizer.decode(token)
                        logging.debug(f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n")
                        choice = {
                            "text": post_processing_text(output, self.task_info["stop"], self.deny_list),
                            "index": beam_id,
                            "finish_reason": "length"+str(sample_id)
                        }
                        if output_scores:
                            choice['logprobs'] = logprobs_buffer[i_output]
                        item['choices'].append(choice)
                        
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": item['choices'],
                "raw_compute_time": time_elapsed,
                # "output_length": [outputs.sequences.shape[0] for outputs in output_buffer]
            }

        #if self.task_info["logprobs"] > 0:
        #    result['logprobs'] = logprobs
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--together_model_name', type=str, default=os.environ.get('SERVICE', 'Together-gpt-JT-6B-v1'),
                        help='worker name for together coordinator.')
    parser.add_argument('--hf_model_name', type=str, default='facebook/opt-350m',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--model_path', type=str, default='',
                        help='hugging face model path (used to load config).')
    parser.add_argument('--worker_name', type=str, default=os.environ.get('WORKER', 'worker1'),
                        help='worker name for together coordinator.')
    parser.add_argument('--group_name', type=str, default=os.environ.get('GROUP', 'group1'),
                        help='group name for together coordinator.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='batch inference, the max batch for .')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device.')
    parser.add_argument('--dtype', type=str, default="",
                        help='dtype.')
    parser.add_argument('--plugin', type=str, default="",
                        help='plugin.')
    args = parser.parse_args()

    plugin = None
    if args.plugin == "faiss_retrieval":
        plugin = FaissRetrievalPlugin()

    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coord_http_port = os.environ.get("COORD_HTTP_PORT", "8092")
    coord_ws_port = os.environ.get("COORD_WS_PORT", "8093")
    deny_list = []
    try:
        deny_list = json.loads(os.environ.get("DENY_LIST", "[]"))
    except Exception as e:
        logging.error(f"failed to parse deny list: {e}")
    try:
        deny_list_file = os.environ.get("DENY_LIST_FILE", "")
        if deny_list_file != None:
            with open(deny_list_file, "r") as f:
                deny_list = [line.strip() for line in f.readlines()]
    except Exception as e:
        logging.error(f"failed to parse deny list file: {e}")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:{coord_http_port}",
        websocket_url=f"ws://{coord_url}:{coord_ws_port}/websocket"
    )
    fip = HuggingFaceLocalNLPModelInference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "device": args.device,
        "dtype": torch_dtype_from_dtype(args.dtype) if args.dtype else None,
        "hf_model_name": args.hf_model_name,
        "model_path": args.model_path,
        "worker_name": args.worker_name,
        "group_name": args.group_name,
        "max_batch_size": args.max_batch_size,
        "gpu_num":1,
        "gpu_type":"RTX 3090",
        "gpu_mem":2400000,
        "deny_list": deny_list,
        "plugin": plugin,
    })
    fip.start()
