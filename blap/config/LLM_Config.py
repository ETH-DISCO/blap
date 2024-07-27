import json

class LLM_Config:
    def __init__(self, jsonData):
        self.t5_model: str = jsonData["t5_model"]
        self.repetition_penalty: str = jsonData["repetition_penalty"]

    @classmethod
    def from_file(cls, jsonFile):
        with open(jsonFile, "r") as f:
            data = json.load(f)
        return cls(data)