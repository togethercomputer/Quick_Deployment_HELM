import os
import random
import math
import timeit
from typing import Dict
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions
import logging
from glm_utils import *
from utils import *


class DistGLMInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        print(f"Model name: {model_name}")
        print("\n=============== Arguments ===============")
        print(args.keys())
        print(args)
        print("=========================================\n")
        self.task_info = {
            "seed": 0,
            "prompt_seqs": None,
            "max_tokens": 16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "temperature": 0.1,
            "stop": [],
            "logprobs": 0,
        }

        model, tokenizer = initialize_model_and_tokenizer(args)

        self.model = model
        self.tokenizer = tokenizer
        self.end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        if dist.get_rank() == 0:
            print(f"<DistGLMInference.__init__> initialization done")

    def dispatch_request(self, args, env) -> Dict:
        print(f"<DistGLMInference.dispatch_request> starts")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["seed"] = get_int(args.get("seed", 0), default=0)
        self.task_info["prompt_seqs"] = [str(args['prompt'])]
        self.task_info["max_tokens"] = get_int(args.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(args.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(args.get("top_k", 50), default=50)
        self.task_info["top_p"] = get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["temperature"] = get_float(args.get("temperature", 0.8), default=0.8)
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["logprobs"] = get_int(args.get("logprobs", 0), default=0)

        if len(self.task_info["prompt_seqs"][0]) == 0 or self.task_info["max_tokens"] == 0:
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
            print(f"<DistGLMInference.dispatch_request> (empty input or output) return: {result}")
            return result
        else:
            result = self._run_inference()
            torch.cuda.empty_cache()
            print(f"<DistGLMInference.dispatch_request> return: {result}")
            return result

    def _run_inference(self):
        print(f"<DistGLMInference._run_inference> start.")
        print(f"Rank-<{dist.get_rank()}> join inference.")
        start_time = time.time()
        raw_text = self.task_info['prompt']
        for i in range(len(raw_text)):
            raw_text[i] = raw_text[i].strip()

        batch_size = min(len(raw_text), 32)
        num_iter = math.ceil(len(raw_text) / batch_size)
        answers = []
        last_layer_embedding = []
        top_logprobs = []
        if self.task_info['seed'] is not None:
            torch.manual_seed(self.task_info['seed'])
            np.random.seed(self.task_info['seed'])
            random.seed(self.task_info['seed'])
            # if debug_print:
            print(f"<DistGLMInference> Rank-<{dist.get_rank()}> setup random seed: {self.task_info['seed']}")

        for iter_i in range(num_iter):
            current_raw_text = raw_text[iter_i * batch_size: (iter_i + 1) * batch_size]
            if self.task_info['temperature'] == 0:
                strategy = BaseStrategy(batch_size=len(current_raw_text), temperature=1, top_k=1,
                                        top_p=self.task_info['top_p'], end_tokens=self.end_tokens)
            else:
                strategy = BaseStrategy(batch_size=len(current_raw_text), temperature=self.task_info['temperature'],
                                        top_k=self.task_info['top_k'], top_p=self.task_info['top_p'],
                                        end_tokens=self.end_tokens)

            cur_answer, cur_last_layer_embedding, cur_top_logprobs = fill_blanks_efficient(current_raw_text,
                                                                                           self.model, self.tokenizer,
                                                                                           strategy, self.task_info)
            answers.extend(cur_answer)
            if cur_last_layer_embedding is None:
                last_layer_embedding = None
            else:
                last_layer_embedding.extend(cur_last_layer_embedding)
            if cur_top_logprobs is None:
                top_logprobs = None
            else:
                top_logprobs.extend(cur_top_logprobs)
            if dist.get_rank() == 0:
                print(f"<DistGLMInference> Current iter handled: {len(answers)}/{len(raw_text)}")

        end_time = time.time()
        if dist.get_rank() == 0:
            prompt_str_lengths = []
            for text in raw_text:
                prompt_str_lengths.append(len(text))
            result = to_result(answers, self.task_info, prompt_str_lengths, top_logprobs)
            return_payload = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": result,
                'raw_compute_time': end_time - start_time
            }
            return return_payload
        else:
            return None

    def _sync_task_info(self):
        print(f"<DistGLMInference._sync_task_info> enter rank-<{dist.get_rank()}>")
        dist.barrier()
        if dist.get_rank() == 0:
            dist.broadcast_object_list([self.task_info], src=0)
        else:
            info = [None]
            dist.broadcast_object_list(info, src=0)
            self.task_info = info[0]
        dist.barrier()
        print(f"<DistGLMInference._sync_task_info> leave rank-<{dist.get_rank()}, task_info:{self.task_info}>")

    def worker(self):
        while True:
            self._sync_task_info()
            self._run_inference()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    args = initialize()
    fip = DistGLMInference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "model_path": args.model_path,
        "worker_name": args.worker_name,
        "group_name": args.group_name,
    })
    fip.start()
