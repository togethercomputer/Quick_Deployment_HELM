import torch


def get_int(input_: str, default=0) -> int:
    try:
        my_num = int(input_)
        return my_num
    except ValueError:
        print(f'Invalid int {input_} set to default: {default}')
        return default


def get_float(input_: str, default=0.0) -> float:
    try:
        my_num = float(input_)
        return my_num
    except ValueError:
        print(f'Invalid float {input_} set to default: {default}')
        return default


def post_processing_text(output_text, stop_tokens):
    print(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)

    print(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    print(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    print(f"<post_processing_text>2 end_pos: {end_pos}.")
    print(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


def convert_hf_score_to_logprobs(scores, k, tokenizer):
    logprobs = []
    for current_step_score in scores:
        print("score shape: ", current_step_score.shape)
        print("score max: ", current_step_score.max())
        value, indices = torch.topk(torch.log_softmax(torch.squeeze(current_step_score.float()), dim=-1), k)
        current_logprob = list(zip(tokenizer.convert_ids_to_tokens(indices.tolist()), value.tolist()))
        logprobs.append(current_logprob)
    return logprobs
