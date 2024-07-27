import json

from blap.config.AudioEncoder_Config import AudioEncoder_Config
from blap.config.LLM_Config import LLM_Config

class BLAP2_Stage1_Config:
    def __init__(self, jsonData):
        self.audio_encoder: AudioEncoder_Config = AudioEncoder_Config(jsonData["audio_encoder"])
        self.num_query_tokens = jsonData["num_query_tokens"]
        self.embed_dim = jsonData["embed_dim"]
        self.max_txt_len = jsonData["max_txt_len"]
    
    @classmethod
    def from_file(cls, jsonFile):
        with open(jsonFile, "r") as f:
            data = json.load(f)
        return cls(data)
    
class BLAP2_Stage2_Config:
    def __init__(self, jsonData):
        self.audio_encoder: AudioEncoder_Config = AudioEncoder_Config(jsonData["audio_encoder"])
        self.LLM: LLM_Config = LLM_Config(jsonData["LLM"])
        self.num_query_tokens: int = jsonData["num_query_tokens"]
        self.embed_dim: int = jsonData["embed_dim"]
        self.max_txt_len: int = jsonData["max_txt_len"]
        self.prompt: str = jsonData["prompt"]
        self.apply_lemmatizer: bool = jsonData["apply_lemmatizer"]
        self.qFormer_ckpt: str = jsonData["qFormer_ckpt"] if "qFormer_ckpt" in jsonData.keys() else ""
        self.qTokens: str = jsonData["qTokens"] if "qTokens" in jsonData.keys() else ""
        self.atRandom: bool = jsonData["atRandom"] if "atRandom" in jsonData.keys() else False
        self.ln_audio: str = jsonData["ln_audio"] if "ln_audio" in jsonData.keys() else ""
        
    @classmethod
    def from_file(cls, jsonFile):
        with open(jsonFile, "r") as f:
            data = json.load(f)
        return cls(data)