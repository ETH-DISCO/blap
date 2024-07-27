# 
#   The Source Code is based on BLIP from SalesForce.com 
#   Date: 24.08.2023
#   Edited by: Constantin Pinkl
# 

from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import nn
from sklearn import logger
import json
import torch
import torch.nn.functional as F
from typing import Any, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from blap.config.config import Config
from blap.dataset.dataset import MusicCaps, ShutterStock
from blap.model.AudioEncoder import AudioEncoder
from blap.model.BLAP import init_tokenizer
from blap.model.BLAP.med import BertConfig, BertModel, BertLMHeadModel
from blap.model.AudioEncoder.AudioEncoder import Projection


class BLAP_Pretrain(pl.LightningModule):

    @classmethod
    def from_checkpoint(cls, config: str, train_config: str, ckpt: str):
        model: BLAP_Pretrain = cls(config, train_config)
        if not torch.cuda.is_available():
            weights = torch.load(ckpt, map_location=torch.device('cpu'))
        else: 
            weights = torch.load(ckpt)
        model.load_state_dict(weights)

        return model

    def __init__(self, config: str, train_config: str) -> None:
        super().__init__()
        with open(config, "r") as f:
            model_cfg = json.load(f)

        with open(train_config, "r")as f:
            self.trainConfig = json.load(f)

        config = Config(model_cfg)
        audio_enc_config = Config(config.audio_encoder)
        audio_cfg = Config(audio_enc_config.audio_cfg)
        bert_cfg = config.bert_cfg
        embed_dim_audio = audio_enc_config.embed_dim  # By default assumed to be 1024
        embed_dim = config.embed_dim           # By default assumed to be 256

        # Create Audio Encoder 
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim_audio, audio_cfg=audio_cfg, joint_embed_shape=audio_enc_config.joint_dim)

        # Create Momentum Audio Encoder
        self.audio_encoder_m = AudioEncoder(embed_dim=embed_dim_audio, audio_cfg=audio_cfg, joint_embed_shape=audio_enc_config.joint_dim)

        # Restore AudioEncoder Weights if applicable
        if hasattr(audio_enc_config, 'pretrained'):
            weights = torch.load(audio_enc_config.pretrained)
            self.audio_encoder.load_state_dict(weights)
            self.audio_encoder_m.load_state_dict(weights)
        # Create Text Encoder
        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_dict(bert_cfg)
        encoder_config.encoder_width = embed_dim_audio
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", config=encoder_config, add_pooling_layer=False)

        if "pretrained" in bert_cfg.keys():
            weights = torch.load(bert_cfg["pretrained"])
            self.text_encoder.load_state_dict(weights, strict=False) # Required strict=False as pretrained model does not have cross attention and has pooling layer

        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 

        text_width = self.text_encoder.config.hidden_size

        # Create Momentum Text Encoder
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)

        # Had to create another Linear Layer on top to adjust dim for both embeddings
        self.audio_proj = Projection(embed_dim_audio, embed_dim)
        self.text_proj =  Projection(text_width, embed_dim)

        # Set up Text Proj
        if "pretrained_text_proj" in model_cfg.keys():
            weights = torch.load(model_cfg["pretrained_text_proj"])
            self.text_proj.load_state_dict(weights)

        # Set up Audio Proj
        if "pretrained_audio_proj" in model_cfg.keys():
            weights = torch.load(model_cfg["pretrained_audio_proj"])
            self.audio_proj.load_state_dict(weights)

        # Projection Momentum
        self.audio_proj_m = Projection(embed_dim_audio, embed_dim)
        self.text_proj_m = Projection(text_width, embed_dim)
        
        # Linear Layer creating binary Audio Text Matching Head
        self.atm_head = nn.Linear(text_width, 2) 

        # Pairing Momentum with Non-Momentum
        self.model_pairs = [[self.audio_encoder,self.audio_encoder_m],
                            [self.audio_proj,self.audio_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]

        self.copy_params()

        # create the queue
        self.register_buffer("audio_queue", torch.randn(embed_dim, config.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, config.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.audio_queue = nn.functional.normalize(self.audio_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = config.queue_size
        self.momentum = config.momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))  
        
        # Create Text Decoder
        decoder_config = BertConfig.from_dict(bert_cfg)
        decoder_config.encoder_width = embed_dim_audio
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config) 

        if "pretrained" in bert_cfg.keys():
            weights = torch.load(bert_cfg["pretrained"])
            pretrained_dict = {f"bert.{k}": v for k, v in weights.items() if f"bert.{k}" in self.text_decoder.state_dict()}
            weights.update(pretrained_dict)
            self.text_decoder.load_state_dict(weights, strict=False) # Required strict=False as pretrained model does not have cross attention and has pooling layer

        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 

        # Tie Encoder Decoder Model Together
        tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')

        # Freeze AudioEncoder
        if hasattr(audio_enc_config, 'freeze') and audio_enc_config.freeze:
            print("Freezing Audio Encoder")
            self.freezeAudioEncoder(self.audio_encoder)
            self.freezeAudioEncoder(self.audio_encoder_m)

        # Loss History
        self.validation_step_outputs = []


    def get_embedding_similarity(self, audio, caption, device=None):
        """
        Returns Cosine Similarity between the two embedding vectors of audio and caption
        """
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

            if device == None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Forward Audio through Audio Encoder
            audio_enc = self.audio_encoder(audio, device)
            audio_embeds = audio_enc[0]
            audio_feat = F.normalize(self.audio_proj(audio_embeds), dim=-1)

            # Text Embedding
            text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=60, 
                                return_tensors="pt").to(device)  
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

            sim = audio_feat @ text_feat.T
        return sim, audio_feat, text_feat
    
    def get_matching_score(self, audio, captions, device=None):
        """
        Compute Matching score for audio-text matching (ATM) loss
        Returned as softmax tensor
        """

        # Fix device if not provided
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            # Compute Audio Emebedding
            audio_enc = self.audio_encoder(audio, device)
            audio_embeds = audio_enc[0]
            audio_atts = torch.ones(audio_embeds.size()[0], 1, dtype=torch.long).to(device)

            # Compute Caption-Audio Embedding
            text = self.tokenizer(captions, padding='max_length', truncation=True, max_length=60, 
                                return_tensors="pt").to(device)  
            
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:,0] = self.tokenizer.enc_token_id

            output_matching_encoder = self.text_encoder(
                                        encoder_input_ids,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = audio_embeds.unsqueeze(1), # Expected batch_size x sequences x Encoder Width
                                        encoder_attention_mask = audio_atts,               # Passing Ones Vector to attend to every hidden state
                                        return_dict = True,
                                    )
            embed = output_matching_encoder.last_hidden_state[:,0,:]

            # Pass through linear layers to obtain matchting score
            vl_output = self.atm_head(embed)

        # return with softmax function
        raise torch.nn.functional.softmax(vl_output, dim=1)
    
    def greedy_decode(self, audio_embeds, audio_atts, max_length=100):
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(audio_embeds.device)  # Start with <BOS> token

        for _ in range(max_length):
            decoder_output = self.text_decoder(input_ids,                             
                                            attention_mask = torch.ones(input_ids.shape, device=input_ids.device), 
                                            encoder_hidden_states = audio_embeds.unsqueeze(1),
                                            encoder_attention_mask = audio_atts,                
                                            return_dict = True)  
            
            # Get the token ID for the token with the highest probability
            next_token_id = decoder_output.logits[:,-1,:].argmax(dim=-1).unsqueeze(-1)
            
            # Check for <EOS> token or simply append the next_token_id to the current sequence
            if next_token_id[0][0] == self.tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        return input_ids

    def get_caption(self, audio, max_length=100, device=None):
        """
        Generate caption for predefined max_length. Audio must be provided in torch tensor format
        """
        
        # Fix device if not provided
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            # Get Audio Embedding from decoder
            audio_enc = self.audio_encoder(audio, device)
            audio_embeds = audio_enc[0]
            audio_atts = torch.ones(audio_embeds.size()[0], 1, dtype=torch.long).to(device)

            # Use Decoder to generate tokens
            input_ids = self.greedy_decode(audio_embeds=audio_embeds, audio_atts=audio_atts, max_length=max_length)

            # Convert token IDs to text
            text_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return text_output

    def forward(self, audio, captions, alpha, device=None):

        assert audio.shape[0] == len(captions), "Amount of captions must equal amount of audio"
        assert len(captions) > 1, "At least two audio, caption pairs required"

        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Forward Audio through Audio Encoder
        audio_enc = self.audio_encoder(audio, device) # Default width 512 ()
        audio_embeds = audio_enc[0]
        audio_atts = torch.ones(audio_embeds.size()[0], 1, dtype=torch.long).to(device)
        # audio_feat = audio_enc[1]
        audio_feat = F.normalize(self.audio_proj(audio_embeds))

        # Text Embedding
        text = self.tokenizer(captions, padding='max_length', truncation=True, max_length=60, 
                              return_tensors="pt").to(device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

        # Momentum Features
        with torch.no_grad():
            # Audio
            self._momentum_update()
            audio_enc_m = self.audio_encoder_m(audio, device=device)
            audio_feat_m = F.normalize(self.audio_proj(audio_enc_m[0]))
            audio_feat_all = torch.cat([audio_feat_m.t(),self.audio_queue.clone().detach()],dim=1)    

            # Text
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')  
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            # As we already normed we can simply use matrix product
            sim_a2t_m = audio_feat_m @ text_feat_all / self.temp 
            sim_t2a_m = text_feat_m @ audio_feat_all / self.temp 

            sim_targets = torch.zeros(sim_a2t_m.size()).to(device)
            sim_targets.fill_diagonal_(1)

            sim_a2t_targets = alpha * F.softmax(sim_a2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2a_targets = alpha * F.softmax(sim_t2a_m, dim=1) + (1 - alpha) * sim_targets 
        
        # As we already normed we can simply use matrix product
        sim_a2t = audio_feat @ text_feat_all / self.temp
        sim_t2a = text_feat @ audio_feat_all / self.temp
        
        ###============== Audio-text Contrastive Loss ===================###

        loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1)*sim_a2t_targets,dim=1).mean()
        loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1)*sim_t2a_targets,dim=1).mean() 

        loss_atc = (loss_a2t + loss_t2a) / 2

        # Add to queue
        self._dequeue_and_enqueue(audio_feat_m, text_feat_m) 

        ###================ Audio-text Matching Loss ====================###

        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positve audio-text pair
        bs = len(audio)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = audio_embeds.unsqueeze(1), # Expected batch_size x sequences x Encoder Width
                                       encoder_attention_mask = audio_atts,               # Passing Ones Vector to attend to every hidden state
                                       return_dict = True,
                                      )
        
        with torch.no_grad():       
            weights_t2a = F.softmax(sim_t2a[:,:bs],dim=1)+1e-4 
            weights_t2a.fill_diagonal_(0)            
            weights_a2t = F.softmax(sim_a2t[:,:bs],dim=1)+1e-4  
            weights_a2t.fill_diagonal_(0)

        # select a negative audio for each text
        audio_emebed_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2a[b], 1).item()
            audio_emebed_neg.append(audio_embeds[neg_idx])
        audio_emebed_neg = torch.stack(audio_emebed_neg,dim=0)   
        audio_emebed_neg = audio_emebed_neg.unsqueeze(1)

        # select a negative text for each audio
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_a2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        audio_emebes_all = torch.cat([audio_emebed_neg,audio_embeds.unsqueeze(1)],dim=0)
        audio_atts_all = torch.cat([audio_atts,audio_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = audio_emebes_all,
                                       encoder_attention_mask = audio_atts_all,      
                                       return_dict = True,
                                      )    
        
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.atm_head(vl_embeddings)            

        atm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(device)
        weights = torch.tensor([3.0, 2.0])
        weights = weights.to(device)
        loss_atm = F.cross_entropy(vl_output, atm_labels, weight=weights)

        ##========== Audio Grounded Language Modeling =============##

        decoder_input_ids = text.input_ids.clone()      
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 

        decoder_output = self.text_decoder(decoder_input_ids,                                    # Output BatchSize x MaxTokenLength x Vocabsize
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = audio_embeds.unsqueeze(1),
                                           encoder_attention_mask = audio_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
          
        loss_lm = decoder_output.loss  

        ##========== Return Triplet Loss ================##
        return (loss_atc, loss_atm, loss_lm)
    
    # Adding necessary pytorch_lightning methods here

    def setup(self, stage=None) -> None:
        trainSetConfig = self.trainConfig["trainSet"]
        set_type = trainSetConfig["type"]
        dataset = ShutterStock(trainSetConfig['data'], trainSetConfig['music']) if set_type == 'ShutterStock' else MusicCaps(trainSetConfig['data'], trainSetConfig['music'])
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.trainConfig["learn_rate"])

    def training_step(self, batch, batch_idx):
        audio, caption = batch
        loss_atc, loss_atm, loss_lm = self(audio, caption, self.trainConfig["alpha"])

        # Logging
        self.log('loss_atc_train', loss_atc.item())
        self.log('loss_atm_train', loss_atm.item())
        self.log('loss_lm_train', loss_lm.item())
        self.log("loss_train", loss_lm.item() + loss_atc.item() + loss_atm.item())
        return loss_atc + loss_atm + loss_lm

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.trainConfig["batch_size"], shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.trainConfig["batch_size"], shuffle=True, drop_last=True)

    def validation_step(self, batch, batch_idx):
        audio, caption = batch
        loss_atc, loss_atm, loss_lm = self(audio, caption, self.trainConfig["alpha"])
        
        # Logging
        self.log('loss_atc_val', loss_atc.item())
        self.log('loss_atm_val', loss_atm.item())
        self.log('loss_lm_val', loss_lm.item())
        self.log("val_loss", loss_lm.item() + loss_atc.item() + loss_atm.item())
        self.validation_step_outputs.append( {'loss_atc_val': loss_atc.item(), 'loss_atm_val': loss_atm.item(), "loss_lm_val": loss_lm.item() + loss_atc.item() + loss_atm.item()})

    def on_validation_epoch_end(self):
        avg_atc = np.stack([x['loss_atc_val'] for x in self.validation_step_outputs]).mean()
        avg_atm = np.stack([x['loss_atm_val'] for x in self.validation_step_outputs]).mean()
        avg_lm = np.stack([x['loss_lm_val'] for x in self.validation_step_outputs]).mean()

        self.log('avg_atc', avg_atc)
        self.log('avg_atm', avg_atm)
        self.log('avg_lm', avg_lm)
        self.log('avg_loss', avg_atc + avg_atm + avg_lm)
        self.validation_step_outputs = []
        return {'avg_atc': avg_atc, 'avg_atm': avg_atm, "avg_lm": avg_lm, "avg_loss": avg_atc + avg_atm + avg_lm}
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, audio_feat, text_feat):
        batch_size = audio_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.audio_queue[:, ptr:ptr + batch_size] = audio_feat.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 

    def freezeAudioEncoder(self, audioEncoder: AudioEncoder):
        for parameter in audioEncoder.parameters():
            parameter.requires_grad_(False)

def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  
