import requests

class RetrievalPlugin():
    def __init__(self, url=None):
        self.url = url if url else "http://host.docker.internal:5001"
        print("RetrievalPlugin init", self.url)

    def request(self, args, env, state):
        # print("RetrievalPlugin request", args)
        req = { "prompt": args[0]["prompt"] }
        result = requests.post(self.url, json=req).json()
        choice = result["data"]["choices"][0]
        state["passage"] = choice["passage"]
        args[0]["prompt"] = choice["text"]

    def response(self, result, state):
        # print("RetrievalPlugin response", result)
        passage = state["passage"]
        choice = result["choices"][0]
        res = choice["text"]
        choice["text"] = f"{res}\n\nPassage:\n{passage}"
        return result
