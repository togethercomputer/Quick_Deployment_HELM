import requests

class FaissRetrievalPlugin():
    def __init__(self, url="http://host.docker.internal:5001"):
        print("FaissRetrievalPlugin init", url)
        self.url = url

    def request(self, args, env, state):
        # print("FaissRetrievalPlugin request", args)
        req = { "prompt": args[0]["prompt"] }
        result = requests.post(self.url, json=req).json()
        choice = result["data"]["choices"][0]
        state["passage"] = choice["passage"]
        args[0]["prompt"] = choice["text"]

    def response(self, result, state):
        # print("FaissRetrievalPlugin response", result)
        passage = state["passage"]
        choice = result["choices"][0]
        res = choice["text"]
        choice["text"] = f"{res}\n\nPassage:\n{passage}"
        return result
