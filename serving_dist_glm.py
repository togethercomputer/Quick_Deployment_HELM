import logging
import math
import os
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherClientOptions, TogetherWeb3
from together_worker.fast_inference import FastInferenceInterface

import utils
from glm_utils import (
    BaseStrategy,
    fill_blanks_efficient,
    initialize,
    initialize_model_and_tokenizer,
    to_result,
)


class DistGLMInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None, glm_args=None) -> None:
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend="mpi")
        except Exception:
            print("[INFO] WARNING: Have initialized the process group")
        args["worker_name"] = "worker" + str(dist.get_rank())
        args["workers"] = dist.get_world_size()
        args["rank"] = dist.get_rank()
        super().__init__(model_name, args if args is not None else {})

        print(f"Model name: {model_name}")
        print("\n=============== <DistGLMInference> Arguments ===============")
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
            "max_sequence_length": glm_args.max_sequence_length,
            "prompt_embedding": False,
        }

        model, tokenizer = initialize_model_and_tokenizer(glm_args)

        self.model = model
        self.tokenizer = tokenizer
        self.end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        if dist.get_rank() == 0:
            print("<DistGLMInference.__init__> initialization done")

    def dispatch_request(self, args, env) -> Dict:
        print("<DistGLMInference.dispatch_request> starts")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["seed"] = utils.get_int(args.get("seed", 0), default=0)
        self.task_info["prompt_seqs"] = [str(args["prompt"])]
        self.task_info["max_tokens"] = utils.get_int(
            args.get("max_tokens", 16), default=16
        )
        self.task_info["beam_width"] = utils.get_int(
            args.get("beam_width", 1), default=1
        )
        self.task_info["top_k"] = utils.get_int(args.get("top_k", 50), default=50)
        self.task_info["top_p"] = utils.get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["temperature"] = utils.get_float(
            args.get("temperature", 0.8), default=0.8
        )
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["logprobs"] = utils.get_int(args.get("logprobs", 0), default=0)

        if (
            len(self.task_info["prompt_seqs"][0]) == 0
            or self.task_info["max_tokens"] == 0
        ):
            inference_result = []
            item = {
                "choices": [],
            }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {"text": "", "index": beam_id, "finish_reason": "length"}
                item["choices"].append(choice)
            inference_result.append(item)
            #  So far coordinator does not support batch.
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inference_result[0]["choices"],
                "raw_compute_time": 0.0,
            }
            print(
                f"<DistGLMInference.dispatch_request> (empty input or output) return: {result}"
            )
            return result
        else:
            self._sync_task_info()
            result = self._run_inference()
            torch.cuda.empty_cache()
            print(f"<DistGLMInference.dispatch_request> return: {result}")
            return result

    def _run_inference(self):
        print("<DistGLMInference._run_inference> start.")
        print(f"Rank-<{dist.get_rank()}> join inference.")
        start_time = time.time()
        raw_text = self.task_info["prompt_seqs"]
        for i in range(len(raw_text)):
            raw_text[i] = raw_text[i].strip()
        print(f"DistGLMInference._run_inference: {raw_text}")
        batch_size = min(len(raw_text), 32)
        num_iter = math.ceil(len(raw_text) / batch_size)
        print(
            f"DistGLMInference._run_inference: batch size: {batch_size}, num iter: {num_iter}"
        )
        answers = []
        last_layer_embedding = []
        top_logprobs = []
        if self.task_info["seed"] is not None:
            torch.manual_seed(self.task_info["seed"])
            np.random.seed(self.task_info["seed"])
            random.seed(self.task_info["seed"])
            # if debug_print:
            print(
                f"<DistGLMInference> Rank-<{dist.get_rank()}> setup random seed: {self.task_info['seed']}"
            )

        for iter_i in range(num_iter):
            current_raw_text = raw_text[iter_i * batch_size : (iter_i + 1) * batch_size]
            print(
                f"DistGLMInference._run_inference: current_raw_text: {current_raw_text}"
            )
            if self.task_info["temperature"] == 0:
                strategy = BaseStrategy(
                    batch_size=len(current_raw_text),
                    temperature=1,
                    top_k=1,
                    top_p=self.task_info["top_p"],
                    end_tokens=self.end_tokens,
                )
            else:
                strategy = BaseStrategy(
                    batch_size=len(current_raw_text),
                    temperature=self.task_info["temperature"],
                    top_k=self.task_info["top_k"],
                    top_p=self.task_info["top_p"],
                    end_tokens=self.end_tokens,
                )

            (
                cur_answer,
                cur_last_layer_embedding,
                cur_top_logprobs,
            ) = fill_blanks_efficient(
                current_raw_text, self.model, self.tokenizer, strategy, self.task_info
            )
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
                print(
                    f"<DistGLMInference> Current iter handled: {len(answers)}/{len(raw_text)}"
                )

        end_time = time.time()
        if dist.get_rank() == 0:
            prompt_str_lengths = []
            for text in raw_text:
                prompt_str_lengths.append(len(text))
            result = to_result(
                answers, self.task_info, prompt_str_lengths, top_logprobs
            )
            return_payload = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": result["inference_result"][0]["choices"],
                "raw_compute_time": end_time - start_time,
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
        print(
            f"<DistGLMInference._sync_task_info> leave rank-<{dist.get_rank()}, task_info:{self.task_info}>"
        )

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
        websocket_url=f"ws://{coord_url}:8093/websocket",
    )
    glm_args = initialize()
    print("\n=============== <Main> Arguments ===============")
    for arg, value in sorted(vars(glm_args).items()):
        print(f"{arg}: {value}")
    fip = DistGLMInference(
        model_name="together/glm-130b",
        args={
            "coordinator": coordinator,
            "worker_name": glm_args.worker_name,
            "group_name": glm_args.group_name,
        },
        glm_args=glm_args,
    )
    fip.start()
