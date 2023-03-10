import torch
import logging
from transformers import StoppingCriteria


def list_ends_with(haystack, needle):
    if len(haystack) < len(needle):
        return False
    for i in range(len(needle)):
        if haystack[-1-i] != needle[-1-i]:
            return False
    return True


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer):
        self.stop_ids = []
        for stop_word in stop_words:
            self.stop_ids.append(tokenizer(stop_word).input_ids)
        print("stop_ids", self.stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        output_ids = input_ids.tolist()[0]
        for stop_ids in self.stop_ids:
            print("check if ends with", stop_ids, output_ids)
            if list_ends_with(output_ids, stop_ids):
                print("stopping!!!!")
                return True
        return False


def get_int(input_: str, default=0) -> int:
    try:
        my_num = int(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid int {input_} set to default: {default}')
        return default


def get_float(input_: str, default=0.0) -> float:
    try:
        my_num = float(input_)
        return my_num
    except ValueError:
        logging.debug(f'Invalid float {input_} set to default: {default}')
        return default


def post_processing_text(output_text, stop_tokens, denylist = []):
    logging.debug(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)

    logging.debug(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    logging.debug(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    logging.debug(f"<post_processing_text>2 end_pos: {end_pos}.")
    logging.debug(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    logging.debug(f"<post_processing_text> input: {output_text}")
    logging.debug(f"<post_processing_text> output: {post_processed_text}")
    for word in denylist:
        if post_processed_text.find(word) != -1:
            logging.debug(f"<post_processing_text> denylist word {word} found, set to empty.")
            post_processed_text = "I'm sorry, but I cannot respond to that."
    return post_processed_text


def convert_hf_score_to_logprobs(scores, k, tokenizer):
    results = []
    batch_size = scores[0].shape[0]
    print(f"<convert_hf_score_to_logprobs>: batch size: {batch_size}")

    for i in range(batch_size):
        logprobs = []
        for current_step_score in scores[i:i+1]:
            value, indices = torch.topk(torch.log_softmax(torch.squeeze(current_step_score.float()), dim=-1), k)
            current_logprob = list(zip(tokenizer.convert_ids_to_tokens(indices.tolist()), value.tolist()))
            logprobs.append(current_logprob)
        results.append(logprobs)
    return results


def torch_dtype_from_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        return torch.float32
