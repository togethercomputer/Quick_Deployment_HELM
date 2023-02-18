import os
import sys
import torch
import timeit
from typing import Dict
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions
from transformers import AutoTokenizer, AutoConfig
from torch.nn.utils.rnn import pad_sequence
import argparse
import logging
from utils import *
from model_utils import *
import numpy as np
import random
import math

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(int(os.environ.get('LOG_LEVEL', logging.DEBUG)))


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
        self.device = torch.device('cuda', 0)
        self.hf_model_name = args['hf_model_name']
        self.max_batch_size = args['max_batch_size']
        if args['model_path'] != '':
            model, tokenizer = get_local_huggingface_tokenizer_model(args['hf_model_name'], args['model_path'])
        else:
            model, tokenizer = get_local_huggingface_tokenizer_model(args['hf_model_name'])
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        logging.debug(f"<HuggingFaceLocalNLPModelInference.__init__> initialization done")

    def dispatch_request(self, args, env) -> Dict:
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
            return result
        else:
            result = self._run_inference()
            torch.cuda.empty_cache()
            logging.debug(f"<HuggingFaceLocalNLPModelInference.dispatch_request> return: {result}")
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
            output_scores = self.task_info["logprobs"] > 0
            if output_scores:
                logprobs_buffer = []
            else:
                logprobs_buffer = None

            time = timeit.default_timer()
            for iter_i in range(num_iter):
                contexts = complete_contexts[iter_i * batch_size: (iter_i + 1) * batch_size]
                inputs = self.tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logging.debug(f"start_ids: length ({inputs.input_ids.shape[0]}) ids: {inputs.input_ids}")
                input_length = inputs.input_ids.shape[1]

                if self.task_info["temperature"] == 0:
                    outputs = self.model.generate(
                        **inputs, do_sample=True, top_p=self.task_info['top_p'],
                        temperature=1.0, top_k=1,
                        max_new_tokens=self.task_info["output_len"],
                        return_dict_in_generate=True,
                        output_scores=output_scores,  # return logit score
                        output_hidden_states=False,  # return embeddings
                    )
                else:
                    outputs = self.model.generate(
                        **inputs, do_sample=True, top_p=self.task_info['top_p'],
                        temperature=self.task_info["temperature"],
                        max_new_tokens=self.task_info["output_len"],
                        return_dict_in_generate=True,
                        output_scores=output_scores,  # return logit score
                        output_hidden_states=False,  # return embeddings
                    )
                if output_scores:
                    logprobs = convert_hf_score_to_logprobs(outputs.scores, self.task_info["logprobs"], self.tokenizer)
                    logprobs_buffer.extend(logprobs)
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
                    "text": post_processing_text(output, self.task_info["stop"]),
                    "index": beam_id,
                    "finish_reason": "length"
                }
                item['choices'].append(choice)
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": item['choices'],
                "raw_compute_time": time_elapsed
            }
        else:
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
                            "text": post_processing_text(output, self.task_info["stop"]),
                            "index": beam_id,
                            "finish_reason": "length"
                        }
                        item['choices'].append(choice)
                    inference_result.append(item)
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inference_result,
                "raw_compute_time": time_elapsed,
                "output_length": [outputs.sequences.shape[0] for outputs in output_buffer]
            }
        if self.task_info["logprobs"] > 0:
            result['logprobs'] = logprobs
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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
    args = parser.parse_args()

    coord_url = os.environ.get("COORD_URL", "127.0.0.1")

    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    fip = HuggingFaceLocalNLPModelInference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "hf_model_name": args.hf_model_name,
        "model_path": args.model_path,
        "worker_name": args.worker_name,
        "group_name": args.group_name,
        "max_batch_size": args.max_batch_size
    })
    fip.start()
