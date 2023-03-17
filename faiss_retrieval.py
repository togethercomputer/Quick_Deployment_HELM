import requests

class FaissRetrievalPlugin():
    def __init__(self):
        print("FaissRetrievalPlugin init") 

    def request(self, args, env, state):
        req = { "prompt": args[0]["prompt"] }
        print("FaissRetrievalPlugin request", req)
        result = requests.post("http://host.docker.internal:5001", req).json()
        print("w0w", result)
        choice = result["choices"][0]
        state["passage"] = choice["passage"]
        args[0]["prompt"] = choice["text"]

    def response(self, result, state):
        print("FaissRetrievalPlugin response", result)
        passage = state["passage"]
        choice = resut["choices"][0]
        res = choice["text"]
        choice["text"] = f"\n\"{passage}\"\n\nAnswer: \n{res}"
        return result
