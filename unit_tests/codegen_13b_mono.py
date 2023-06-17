import requests


prompt = "def hello_world():"
top_p = 1.0
temperature = 0.5
max_new_tokens = 32


my_post_dict = {
    "model": "together/codegen-16B-mono",
    "prompt": prompt,
    "top_p": float(top_p),
    "temperature": float(temperature),
    "max_tokens": int(max_new_tokens),
    "stop": [],
    # "logprobs": 3
}

print(my_post_dict)
response = requests.get(
    "https://staging.together.xyz/api/inference", params=my_post_dict
).json()
print(response)
