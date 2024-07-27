import json

from blap.config.config import Config

class AudioEncoder_Config:
    def __init__(self, jsonData):
        if "pretrained" in jsonData:
            self.pretrained = jsonData["pretrained"]
        self.audio_cfg = Config(jsonData["audio_cfg"])
        self.embed_dim_audio: int = jsonData["embed_dim_audio"]
        # self.class_num: int = jsonData["class_num"]

    @classmethod
    def from_file(cls, jsonFile):
        with open(jsonFile, "r") as f:
            data = json.load(f)
        
        return cls(data)