import os
from typing import Dict
import argparse
import configparser
import timeit
import logging
import numpy as np
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig
from faster_transformer.gptneox import GptNeoX

from utils import *
from model_utils import *


class FastRedPajamaInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        
        args['worker_name'] = 'worker0'
        args['workers'] = 1
        args['rank'] = 0
        args['world_size'] = 1
        
        super().__init__(model_name, args if args is not None else {})
        print("\n=============== Arguments ===============")
        print(args.keys())
        print(args)
        print("=========================================\n")
        self.tensor_para_size = args['tensor_para_size']
        self.pipeline_para_size = 1
        self.max_batch_size = args['max_batch_size']
        self.random_seed_tensor = torch.zeros([self.max_batch_size], dtype=torch.int64)
        self.task_info={
            "prompt_seqs": None,
            "output_len":16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "return_cum_log_probs": 0,
            "return_output_length":0,
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(args['hf_model_path'], use_fast=True)
        config = configparser.ConfigParser()
        config.read(os.path.join(args['ckpt_path'], "config.ini"))
        head_num = int(config.get('gptneox', 'head_num'))
        size_per_head = int(config.get('gptneox', 'size_per_head'))
        vocab_size = int(config.get('gptneox', 'vocab_size'))
        layer_num = int(config.get('gptneox', 'num_layer'))
        rotary_embedding = int(config.get('gptneox', 'rotary_embedding'))
        start_id = int(config.get('gptneox', 'start_id'))
        self.end_id = int(config.get('gptneox', 'end_id'))
        use_gptj_residual = (config.get('gptneox', 'use_gptj_residual') == "1")
        weight_data_type = config.get('gptneox', 'weight_data_type')
        lib_path = args["lib_path"]
        ckpt_path = args['ckpt_path']
        max_seq_len = 2048

        torch.manual_seed(0)
        with torch.no_grad():
            # Prepare model.
            self.redpajama_model = GptNeoX(head_num, size_per_head, 
                                           vocab_size, rotary_embedding,
                                           start_id, self.end_id, layer_num, max_seq_len, 
                                           self.tensor_para_size, self.pipeline_para_size, 
                                           use_gptj_residual, lib_path, 
                                           inference_data_type=weight_data_type, 
                                           weights_data_type=weight_data_type)
            if not self.redpajama_model.load(ckpt_path=ckpt_path):
                print("[WARNING] Checkpoint file not found. Model loading is skipped.")
               
        print(f"<FastRedPajamaInference.__init__> initialization done")

        
    def dispatch_request(self, args, env) -> Dict:
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["prompt_seqs"] = [args['prompt']]
        self.task_info["output_len"] = get_int(args.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(args.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(args.get("top_k", 50), default=50)
        self.task_info["top_p"] = get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(args.get("beam_search_diversity_rate", 0.0), default=0.0)
        self.task_info["temperature"] = get_float(args.get("temperature", 0.8), default=0.1)
        self.task_info["len_penalty"] = get_float(args.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(args.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["stream_tokens"] = args.get("stream_tokens", False)
        self.task_info["return_cum_log_probs"] = args.get("return_cum_log_probs", 0)
        self.task_info["return_output_length"] = args.get("return_output_length", 0)
        self.task_info["stream_tokens"] = args.get("stream_tokens", False)
        
        if len(self.task_info["prompt_seqs"][0]) == 0 or self.task_info["output_len"] == 0:
            inferenece_result = []
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {
                    "text": '',
                    "index": beam_id,
                    "finish_reason": "length"
                }
            item['choices'].append(choice)
            inferenece_result.append(item)
            #  So far coordinator does not support batch. 
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inferenece_result[0]['choices'],
                "raw_compute_time": 0.0
            }
            print(f"<FastRedPajamaInference.dispatch_request> (not FT runs, 0 input or output) return: {result}")
            return result
        else:
            result = self._run_inference()
            print(f"<FastRedPajamaInference.dispatch_request> return: {result}")
            return result

    def _run_inference(self):
        print(f"<FastRedPajamaInference._run_inference> enter")
        
        with torch.no_grad():
            contexts = self.task_info["prompt_seqs"]
            start_ids = [torch.IntTensor(self.tokenizer.encode(c)) for c in contexts]
            start_lengths = [len(ids) for ids in start_ids]
            
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)
            start_lengths = torch.IntTensor(start_lengths)
            print(f"start_ids: length ({start_ids.shape[0]}) ids: {start_ids}")
            
            time = timeit.default_timer()
            max_batch_size = self.max_batch_size
            tokens_batch = self.redpajama_model(
                                    start_ids=start_ids,
                                    start_lengths=start_lengths,
                                    output_len=self.task_info["output_len"],
                                    beam_width=self.task_info["beam_width"],
                                    top_k=self.task_info["top_k"] * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                    top_p=self.task_info["top_p"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    beam_search_diversity_rate=self.task_info["beam_search_diversity_rate"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    temperature=self.task_info["temperature"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    len_penalty=self.task_info["len_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    repetition_penalty=self.task_info["repetition_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    random_seed = self.random_seed_tensor,
                                    return_output_length = self.task_info["return_output_length"],
                                    return_cum_log_probs = self.task_info["return_cum_log_probs"],
                                    request_id=self.served,
                                    stream_tokens_pipe = self.stream_tokens_pipe_w if self.task_info["stream_tokens"] else -1)
            # only a thread (rank 0) gets the output, while the others are supposed to return None.
            time_elapsed = timeit.default_timer() - time
        print("[INFO] Redpajama time costs: {:.2f} ms.>".format(time_elapsed * 1000))
        
        assert tokens_batch is not None
    
        if self.task_info["return_cum_log_probs"] > 0:
            tokens_batch, _, cum_log_probs = tokens_batch
            print('[INFO] Log probs of sentences:', cum_log_probs)

        inferenece_result = []
        tokens_batch = tokens_batch.cpu().numpy()
        
        for i, (context, tokens) in enumerate(zip(self.task_info["prompt_seqs"], tokens_batch)):
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                output = self.tokenizer.decode(token)
                print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")
                choice = {
                    "text": post_processing_text(output, self.task_info["stop"]),
                    "index": beam_id,
                    "finish_reason": "length"
                }
            item['choices'].append(choice)
            inferenece_result.append(item)
        #  So far coordinator does not support batch. 
        return {
            "result_type": RequestTypeLanguageModelInference,
            "choices": inferenece_result[0]['choices'],
            "raw_compute_time": time_elapsed
        }


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--together_model_name', type=str, default=os.environ.get('SERVICE', 'Together-RedPajama-INCITE-7B-Chat'),
                        help='worker name for together coordinator.')
    parser.add_argument('--ckpt_path', type=str, default='/workspace/FasterTransformer/build/model/ft-RedPajama-INCITE-7B-Chat/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--hf_model_path', type=str, default='togethercomputer/RedPajama-INCITE-7B-Chat',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--lib_path', type=str, default='/workspace/FasterTransformer/build/lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--worker_name', type=str, default=os.environ.get('WORKER','worker1'),
                        help='worker name for together coordinator.')
    parser.add_argument('--group_name', type=str, default=os.environ.get('GROUP', 'group1'),
                        help='group name for together coordinator.')
    
    args = parser.parse_args()
    
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coord_http_port = os.environ.get("COORD_HTTP_PORT", "8092")
    coord_ws_port = os.environ.get("COORD_WS_PORT", "8093")

    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:{coord_http_port}",
        websocket_url=f"ws://{coord_url}:{coord_ws_port}/websocket"
    )
    fip = FastRedPajamaInference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "hf_model_path": args.hf_model_path,
        "worker_name": args.worker_name,
        "group_name": args.group_name,
        "ckpt_path": args.ckpt_path,
        "lib_path": args.lib_path,
        "tensor_para_size":args.tensor_para_size,
        "stream_tokens_pipe": False,
        "gpu_num": 1,
        "gpu_type": "A100-40G",
        "gpu_mem": 8000000,
        "max_batch_size": 1
    })
    fip.start()