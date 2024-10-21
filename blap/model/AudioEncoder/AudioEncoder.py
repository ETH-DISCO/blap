"""
Copyright Laion-AI
https://github.com/LAION-AI/CLAP/
"""
import logging
from blap.model.AudioEncoder.acnn import get_audio_encoder
from blap.model.AudioEncoder.htsat import create_htsat_model
import torch
from torch import nn
import torch.nn.functional as F

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(torch.nn.Module):

    def _get_audio_embeddings_CNN(self, preprocessed_audio):
        r"""Load preprocessed audio and return a audio embeddings"""
        with torch.no_grad():
            # preprocessed_audio = preprocessed_audio.reshape(
            #     preprocessed_audio.shape[0], preprocessed_audio.shape[2])
            #Append [0] the audio emebdding, [1] has output class probabilities
            out_dict = self.audio_branch(preprocessed_audio)
            audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
            return audio_features

    def __init__(self,
                 embed_dim: int, 
                 audio_cfg, 
                 mlp_act: str ="relu", 
                 enable_fusion: bool = False, 
                 joint_embed_shape: int = 512,
                 fusion_type='None', 
                 audio_branch=None, audio_projection=None):
        super().__init__()

        self.joint_embed_shape = joint_embed_shape
        if mlp_act == 'relu':
            mlp_act_layer = nn.ReLU()
        elif mlp_act == 'gelu':
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError
        
        self.audio_cfg = audio_cfg
        self.encoder_type = audio_cfg.model_type

        if audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(audio_cfg, enable_fusion, fusion_type)
            self.audio_projection = nn.Sequential(
                nn.Linear(embed_dim, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
            )
        elif audio_cfg.model_type == "Cnn14":
            audio_encoder = get_audio_encoder("Cnn14")
            self.audio_branch = audio_encoder(
                audio_cfg.sampling_rate, audio_cfg.window_size,
                audio_cfg.hop_size, audio_cfg.mel_bins, audio_cfg.fmin, audio_cfg.fmax,
                audio_cfg.classes_num, embed_dim
            )
            self.audio_projection = Projection(embed_dim, joint_embed_shape)
        else:
            logging.error(f"Model config for {audio_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {audio_cfg.model_type} not found.")

        

        if audio_branch != None:
            self.audio_branch = audio_branch
        
        if audio_projection != None:
            self.audio_projection = audio_projection

    def encode_audio(self, audio, device):
        if self.encoder_type == 'Cnn14':
            return {"embedding": self._get_audio_embeddings_CNN(audio)}
        else:
            return self.audio_branch(audio, mixup_lambda=None, device=device)  # mix lambda needs to add

    
    def forward(self, data: torch.Tensor, device=None):
        """
        Returns tuple
        (Embedding, Audio Projection, Encoding including mutliple results provided by the HTS Audio Transformer)

        The three outputs differ in their specified use cases

        1. Embedding can be used any general task that ready require a specific feature extraction
        2. Projection simplifies Embedding even further and already applies normalization is directly used like that in original CLAP
        3. General Output for all other uses cases that don't necessarly require reduced features already
        """
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # input_dict = {}
        # keys = data[0].keys()
        # for k in keys:
        #     input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        
        # assert data.dim() == 2, "expected shape (BatchSize, ClipLength)"
        # assert data.shape[1] == self.audio_cfg.clip_samples, f"Provided Clip Length differes from expected clip length ({self.audio_cfg.clip_samples})"
        encoding = self.encode_audio(data, device=device)
        embedding = encoding["embedding"]
        proj = self.audio_projection(embedding)
        proj = proj/torch.norm(proj, dim=-1, keepdim=True)
        return  embedding, proj, encoding