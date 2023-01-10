import requests

prompt = "Where is Zurich?"
top_p = 1.0
temperature = 0.5
max_new_tokens = 5


my_post_dict = {
    "model": "together/opt-350m",
    "prompt": prompt,
    "top_p": float(top_p),
    "temperature": float(temperature),
    "max_tokens": int(max_new_tokens),
    "stop": [],
    "logprobs": 0
}

print(my_post_dict)
response = requests.get("https://staging.together.xyz/api/inference", params=my_post_dict).json()
print(response)