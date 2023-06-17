import argparse
import os
import random
import timeit
from typing import Dict

import numpy as np
import torch
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherClientOptions, TogetherWeb3
from together_worker.fast_inference import FastInferenceInterface

import model_utils as mu
import utils


class HuggingFaceDistNLPModelInference(FastInferenceInterface):
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
        self.device = torch.device("cuda", 0)
        self.hf_model_name = args["hf_model_name"]
        assert args["model_path"] != ""
        model, tokenizer = mu.get_dist_accelerate_tokenizer_model(
            args["hf_model_name"], args["model_path"]
        )
        self.model = model
        self.tokenizer = tokenizer
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        print("<HuggingFaceDistNLPModelInference.__init__> initialization done")

    def dispatch_request(self, args, env) -> Dict:
        print("<HuggingFaceDistNLPModelInference.dispatch_request> starts")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["seed"] = utils.get_int(args.get("seed", 0), default=0)
        self.task_info["prompt_seqs"] = [str(args["prompt"])]
        self.task_info["output_len"] = utils.get_int(
            args.get("max_tokens", 16), default=16
        )
        self.task_info["beam_width"] = utils.get_int(
            args.get("beam_width", 1), default=1
        )
        self.task_info["top_k"] = utils.get_int(args.get("top_k", 50), default=50)
        if self.hf_model_name == "google/flan-t5-xxl":
            self.task_info["top_p"] = utils.get_float(
                args.get("top_p", 1.0), default=1.0
            )
        else:
            self.task_info["top_p"] = utils.get_float(
                args.get("top_p", 0.0), default=0.0
            )
        self.task_info["beam_search_diversity_rate"] = utils.get_float(
            args.get("beam_search_diversity_rate", 0.0), default=0.0
        )
        self.task_info["temperature"] = utils.get_float(
            args.get("temperature", 0.8), default=0.8
        )
        self.task_info["len_penalty"] = utils.get_float(
            args.get("len_penalty", 0.0), default=0.0
        )
        self.task_info["repetition_penalty"] = utils.get_float(
            args.get("repetition_penalty", 1.0), default=1.0
        )
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["logprobs"] = utils.get_int(args.get("logprobs", 0), default=0)

        if (
            len(self.task_info["prompt_seqs"][0]) == 0
            or self.task_info["output_len"] == 0
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
                f"<HuggingFaceDistNLPModelInference.dispatch_request> (empty input or output) return: {result}"
            )
            return result
        else:
            result = self._run_inference()
            torch.cuda.empty_cache()
            print(
                f"<HuggingFaceDistNLPModelInference.dispatch_request> return: {result}"
            )
            return result

    def _run_inference(self):
        print("<HuggingFaceDistNLPModelInference._run_inference> start.")

        with torch.no_grad():
            torch.manual_seed(self.task_info["seed"])
            np.random.seed(self.task_info["seed"])
            random.seed(self.task_info["seed"])

            contexts = self.task_info["prompt_seqs"]
            inputs = self.tokenizer(contexts, return_tensors="pt").to(self.device)
            print(
                f"start_ids: length ({inputs.input_ids.shape[0]}) ids: {inputs.input_ids}"
            )
            input_length = inputs.input_ids.shape[1]

            print(self.task_info)
            output_scores = self.task_info["logprobs"] > 0

            time = timeit.default_timer()
            if self.task_info["temperature"] == 0:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    do_sample=True,
                    top_p=self.task_info["top_p"],
                    temperature=1.0,
                    top_k=1,
                    max_new_tokens=self.task_info["output_len"],
                    return_dict_in_generate=True,
                    output_scores=output_scores,  # return logit score
                    output_hidden_states=False,  # return embeddings
                )
            else:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    do_sample=True,
                    top_p=self.task_info["top_p"],
                    temperature=self.task_info["temperature"],
                    max_new_tokens=self.task_info["output_len"],
                    return_dict_in_generate=True,
                    output_scores=output_scores,  # return logit score
                    output_hidden_states=False,  # return embeddings
                )
            if output_scores:
                logprobs = utils.convert_hf_score_to_logprobs(
                    outputs.scores, self.task_info["logprobs"], self.tokenizer
                )
            else:
                logprobs = None

            time_elapsed = timeit.default_timer() - time

        print(
            f"[INFO] HuggingFaceDistNLPModelInference time costs: {time_elapsed} ms. "
        )

        inference_result = []
        item = {
            "choices": [],
        }
        for beam_id in range(self.task_info["beam_width"]):
            if self.hf_model_name == "google/flan-t5-xxl":
                token = outputs.sequences[beam_id, :]
            else:
                token = outputs.sequences[
                    beam_id, input_length:
                ]  # exclude context input from the output
            print(f"[INFO] raw token: {token}")
            output = self.tokenizer.decode(token)
            print(
                f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n"
            )
            choice = {
                "text": utils.post_processing_text(output, self.task_info["stop"]),
                "index": beam_id,
                "finish_reason": "length",
            }
            item["choices"].append(choice)
            inference_result.append(item)
        #  So far coordinator does not support batch.
        result = {
            "result_type": RequestTypeLanguageModelInference,
            "choices": inference_result[0]["choices"],
            "raw_compute_time": time_elapsed,
        }
        if self.task_info["logprobs"] > 0:
            result["logprobs"] = logprobs
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--together_model_name",
        type=str,
        default=os.environ.get("SERVICE", "Together-gpt-JT-6B-v1"),
        help="worker name for together coordinator.",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="facebook/opt-350m",
        help="hugging face model name (used to load config).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="hugging face model path (used to load config).",
    )
    parser.add_argument(
        "--worker_name",
        type=str,
        default=os.environ.get("WORKER", "worker1"),
        help="worker name for together coordinator.",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default=os.environ.get("GROUP", "group1"),
        help="group name for together coordinator.",
    )

    args = parser.parse_args()

    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coord_http_port = os.environ.get("COORD_HTTP_PORT", "8092")
    coord_ws_port = os.environ.get("COORD_WS_PORT", "8093")

    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:{coord_http_port}",
        websocket_url=f"ws://{coord_url}:{coord_ws_port}/websocket",
    )
    fip = HuggingFaceDistNLPModelInference(
        model_name=args.together_model_name,
        args={
            "coordinator": coordinator,
            "hf_model_name": args.hf_model_name,
            "model_path": args.model_path,
            "worker_name": args.worker_name,
            "group_name": args.group_name,
        },
    )
    fip.start()
