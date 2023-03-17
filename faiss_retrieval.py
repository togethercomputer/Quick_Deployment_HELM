import requests

class FaissRetrievalPlugin():
    def __init__(self, url="http://host.docker.internal:5001"):
        print("FaissRetrievalPlugin init")
        self.url = url

    def request(self, args, env, state):
        req = { "prompt": args[0]["prompt"] }
        print("FaissRetrievalPlugin request", req)
        result = requests.post(self.url, json=req).json()
        choice = result["data"]["choices"][0]
        state["passage"] = choice["passage"]
        args[0]["prompt"] = choice["text"]

    def response(self, result, state):
        print("FaissRetrievalPlugin response", result)
        passage = state["passage"]
        choice = result["choices"][0]
        res = choice["text"]
        choice["text"] = f"\n\"{passage}\"\n\nAnswer: \n{res}"
        return result
