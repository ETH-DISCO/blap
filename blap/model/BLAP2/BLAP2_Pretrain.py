import logging
from transformers import Blip2QFormerConfig, Blip2QFormerModel, BertTokenizer, BertConfig
import torch
from torch import nn
import torch.distributed as dist
from blap.config.BLAP2_Config import BLAP2_Stage1_Config, BLAP2_Stage2_Config
from blap.dataset.dataset import MusicCaps, ShutterStock
from blap.model.BLAP2.BLAP2Outputs import BlapOutput, BlapIntermediateOutput
from blap.model.BLAP2.QFormer import BertLMHeadModel
import soundfile as sf
import json
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import T5TokenizerFast
from blap.model.BLAP2.modeling_t5 import T5Config, T5ForConditionalGeneration
from torch.utils.data.distributed import DistributedSampler



from blap.model.AudioEncoder import AudioEncoder
from blap.model.AudioEncoder import Projection

# Lavis Utilities

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """

    if not is_dist_avail_and_initialized():
        return tensors
    
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

class BLAP2_Base(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    @classmethod
    def initQformer(cls, num_query_token, audio_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = audio_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


class BLAP2_Stage1(BLAP2_Base):
    """
    This stage deals with the training of the QFormer.
    """
    def __init__(self, config: BLAP2_Stage1_Config, trainConfig, forbidden_words=[], forbidden_drop=0.3) -> None:
        super().__init__()

        # Qualitative Analysis Audio, Caption Tuples
        self.forbidden_words = forbidden_words
        self.forbidden_drop = forbidden_drop

        # Training Configuration File
        self.trainConfig = trainConfig

        # Used for tracking previous validation step results -> compute mean over it after validation
        self.validation_step_outputs = []

        # Setup Audio Encoder
        self.audio_encoder = AudioEncoder(embed_dim=config.audio_encoder.embed_dim_audio, audio_cfg=config.audio_encoder.audio_cfg)
        self.ln_audio = torch.nn.LayerNorm(config.audio_encoder.embed_dim_audio)

        # Freeze Audio Encoder Weights
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # Restore AudioEncoder Weights if applicable
        if hasattr(config.audio_encoder, 'pretrained'):
            weights = torch.load(config.audio_encoder.pretrained)
            self.audio_encoder.load_state_dict(weights)

        # Tokenizer
        self.tokenizer: BertTokenizer = self.init_tokenizer()

        # Setup Q Former
        self.qformer, self.query_tokens = self.initQformer(config.num_query_tokens, config.audio_encoder.embed_dim_audio)
        self.qformer.resize_token_embeddings(len(self.tokenizer))

        # TODO Do not fully understand this part yet
        state_dict = self.qformer.state_dict()
        for name, param in self.qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.audio_proj = Projection(self.qformer.config.hidden_size, config.embed_dim)
        self.text_proj = Projection(self.qformer.config.hidden_size, config.embed_dim)

        self.atm_head = nn.Linear(self.qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = config.max_txt_len

    @classmethod
    def from_json(cls, modelConfig, trainConfig):
        if trainConfig == "":
            train_Config = {}
        else: 
            with open(trainConfig, "r")as f:
                train_Config = json.load(f)
        config = BLAP2_Stage1_Config.from_file(modelConfig)
        return cls(config, train_Config)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, modelConfig, trainConfig):
        # Load configurations from files
        config = BLAP2_Stage1_Config.from_file(modelConfig)
        if trainConfig == "":
            train_Config = {}
        else: 
            with open(trainConfig, "r")as f:
                train_Config = json.load(f)

        # Load the model from the checkpoint
        map_location = None

        if not torch.cuda.is_available():
            print("No GPU found, loading on CPU")
            map_location = torch.device('cpu')
        model = cls.load_from_checkpoint(checkpoint_path, config=config, trainConfig=train_Config, map_location=map_location)

        return model

    
    def get_audioFeatures(self, audio, device=None):
        """
            Deprecated
            Returns the embedding provided by the audio encoder
        """
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_embeds = self.audio_encoder(audio, device) # Default width 512 ()
        return audio_embeds
    
    def get_queryFeatures(self, audio: torch.Tensor):
        audio_embeds = self.ln_audio(F.avg_pool2d( self.audio_encoder(audio)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )
        
        return query_output
    
    @torch.no_grad()
    def get_audioEmbeddings(self, audios, device=None):
        """
            Returns the audio embedding of the Q-Former (Sometimes called audio features)

            -> Returns a BatchxQueryTokensxEmbedding
        """
        
        # Set Device to GPU if available
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_embeds: torch.Tensor = self.ln_audio(F.avg_pool2d( self.audio_encoder(audios)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )

        audio_feats: torch.Tensor = F.normalize(
            self.audio_proj(query_output.last_hidden_state), dim=-1
        )

        return audio_feats
    
    @torch.no_grad()
    def get_captionEmbeddings(self, captions, device=None):
        """
            Returns the text embedding of the Q-Former (Somtimes called text features)

            -> Returns BatchSizexEmbedding
        """
        # Set Device to GPU if available
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        caption_tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.qformer.bert(
            caption_tokens.input_ids,
            attention_mask=caption_tokens.attention_mask,
            return_dict=True,
        )

        # Take CLS token as text embedding
        text_feat: torch.Tensor = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return text_feat


    # only provides the loss for training
    def forward(self, audios, captions, device=None):
        # Get Audio Embeddings
        # Fix device if not provided
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # get first word in captions
        for i, cap in enumerate(captions):
            cap = cap.split(" ")
            if cap[0] in self.forbidden_words:
                if torch.rand(1) < self.forbidden_drop:
                    # remove first  from cap
                    cap = cap[1:]
                    captions[i] = ' '.join(cap)

        audio_embeds: torch.Tensor = self.ln_audio(F.avg_pool2d( self.audio_encoder(audios)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )

        audio_feats: torch.Tensor = F.normalize(
            self.audio_proj(query_output.last_hidden_state), dim=-1
        )

        caption_tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.qformer.bert(
            caption_tokens.input_ids,
            attention_mask=caption_tokens.attention_mask,
            return_dict=True,
        )

        # Take CLS token as text embedding
        text_feat: torch.Tensor = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        #-------------------------------------------------------------------#
        #--------------------Audio Text Contrastive Loss--------------------#
        #-------------------------------------------------------------------#

        audio_feats_all = concat_all_gather(
            audio_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t: torch.Tensor = torch.matmul(
            audio_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # audio-text similarity: aggregate across all query tokens
        sim_a2t, _ = sim_q2t.max(-1)
        sim_a2t = sim_a2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), audio_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-audio similarity: aggregate across all query tokens
        sim_t2a, _ = sim_t2q.max(-1)
        sim_t2a = sim_t2a / self.temp  
        try:
            rank = dist.get_rank()
        except RuntimeError:
            print("Not part of cluster, rank not initalised")
            rank = 0 
        bs = audios.size(0)

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            audios.device
        )

        # compute matchting matrix
        loss_atc = (
            F.cross_entropy(sim_a2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2a, targets, label_smoothing=0.1)
        ) / 2

        #----------------------------------------------------------------#    
        #--------------------Audio Text Matching Loss--------------------#
        #----------------------------------------------------------------#

        text_input_ids_world = concat_all_gather(caption_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(caption_tokens.attention_mask)
        audio_embeds_world = all_gather_with_grad(audio_embeds)

        with torch.no_grad():
            sim_t2a[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_a2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            weights_t2i = F.softmax(sim_t2a, dim=1)
            weights_i2t = F.softmax(sim_a2t, dim=1)

        audio_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            audio_embeds_neg.append(audio_embeds_world[neg_idx])
        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)

        # select a negative text for each audio
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [caption_tokens.input_ids, caption_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg

        text_atts_all = torch.cat(
            [caption_tokens.attention_mask, caption_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_atm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_atm.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        audio_embeds_all = torch.cat(
            [audio_embeds, audio_embeds_neg, audio_embeds], dim=0
        )  # pos, neg, pos

        audio_atts_all = torch.ones(audio_embeds_all.size()[:-1], dtype=torch.long).to(
            device
        )

        output_atm = self.qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_atm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=audio_embeds_all,
            encoder_attention_mask=audio_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_atm.last_hidden_state[:, : query_tokens_atm.size(1), :]
        vl_output = self.atm_head(vl_embeddings)
        logits: torch.Tensor = vl_output.mean(dim=1)

        atm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)


        loss_atm = F.cross_entropy(logits, atm_labels)

        #---------------------------------------------------------------#
        #--------------------Language Modelling Loss--------------------#
        #---------------------------------------------------------------#

        decoder_input_ids = caption_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask = torch.cat([query_atts, caption_tokens.attention_mask], dim=1)
        lm_output = self.qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        
        return BlapOutput(
            intermediate_output=BlapIntermediateOutput(
                atm_logits=logits.detach(),
                atm_labels=atm_labels.detach(),
                vl_embedding=vl_embeddings.detach(),
                audio_embeds=audio_embeds.detach(),
                audio_features=audio_feats.detach(),
                text_embeds=text_feat.detach(),
                decoder_output=lm_output,
                text_features=text_feat
            ),
            loss=loss_atc + loss_atm + loss_lm,
            loss_atc=loss_atc,
            loss_atm=loss_atm,
            loss_lm=loss_lm,
        )
    
    
    #------------------------------------------------#
    #----------------PyTorch Lightning---------------#
    #------------------------------------------------#
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
        loss: BlapOutput = self(audio, caption)

        # Logging
        self.log('loss_atc_train', loss.loss_atc.item())
        self.log('loss_atm_train', loss.loss_atm.item())
        self.log('loss_lm_train', loss.loss_lm.item())
        self.log("loss_train", loss.loss.item())
        return loss.loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.trainConfig["batch_size"], shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.trainConfig["batch_size"], shuffle=False, drop_last=True)

    def validation_step(self, batch, batch_idx):
        audio, caption = batch
        # TODO remove debugging print out
        print(f"Validating Batch {batch_idx}")
        loss: BlapOutput = self(audio, caption)
        
        # Logging
        self.log('loss_atc_val', loss.loss_atc.item(), sync_dist=True)
        self.log('loss_atm_val', loss.loss_atm.item(), sync_dist=True)
        self.log('loss_lm_val', loss.loss_lm.item(), sync_dist=True)
        self.log("val_loss", loss.loss_lm.item() + loss.loss_atc.item() + loss.loss_atm.item(), sync_dist=True)
        self.validation_step_outputs.append( {'loss_atc_val': loss.loss_atc.item(), 'loss_atm_val': loss.loss_atm.item(), "loss_lm_val": loss.loss_lm.item()})

    def on_validation_epoch_end(self):
        avg_atc = np.stack([x['loss_atc_val'] for x in self.validation_step_outputs]).mean()
        avg_atm = np.stack([x['loss_atm_val'] for x in self.validation_step_outputs]).mean()
        avg_lm = np.stack([x['loss_lm_val'] for x in self.validation_step_outputs]).mean()
        # TODO remove debugging print out
        print(len(self.validation_step_outputs))

        # Qualitative Assesment of Data 
        # if len(self.qualitativeTestingSamples) > 0 : 
        #     audioToEval = torch.zeros((len(self.qualitativeTestingSamples), 480000))
        #     correspondingCaptions = []
        #     for i, audioCaptionTuple in enumerate(self.audios):
        #         data, _ = sf.read(audioCaptionTuple[0])
        #         assert len(data) == 480000, "Audios must have length of 480000"
        #         audioToEval[i] = data
        #         correspondingCaptions.append(audioCaptionTuple[1])
            
        #     # Forward it through the work
        #     output: BlapOutput = self(audioToEval, correspondingCaptions)
        #     intermediate = output.intermediate_output

        #     # Write to file
        #     # Hard coded file for embeddings + logits of atm
            

        self.log('avg_atc', avg_atc, sync_dist=True)
        self.log('avg_atm', avg_atm, sync_dist=True)
        self.log('avg_lm', avg_lm, sync_dist=True)
        self.log('avg_loss', avg_atc + avg_atm + avg_lm, sync_dist=True)
        self.validation_step_outputs = []
        return {'avg_atc': avg_atc, 'avg_atm': avg_atm, "avg_lm": avg_lm, "avg_loss": avg_atc + avg_atm + avg_lm}
    
# T5 text encoder / decoder model
class BLAP2_Stage2(BLAP2_Base):
    def __init__(self, config: BLAP2_Stage2_Config, trainConfig, forbidden_words=["Upbeat", "upbeat", "Upbeat,", "upbeat,", "The low quality recording", "This is a low-quality"], forbidden_drop=0.6):
        super().__init__()

        self.forbidden_words = forbidden_words
        self.forbidden_drop = forbidden_drop

        self.trainConfig = trainConfig

        self.tokenizer: BertTokenizer = self.init_tokenizer()
        # Setup Audio Encoder
        self.audio_encoder = AudioEncoder(embed_dim=config.audio_encoder.embed_dim_audio, audio_cfg=config.audio_encoder.audio_cfg)
        self.ln_audio = torch.nn.LayerNorm(config.audio_encoder.embed_dim_audio)

        # Freeze Audio Encoder Weights
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # Restore AudioEncoder Weights if applicable
        if hasattr(config.audio_encoder, 'pretrained') and config.audio_encoder.pretrained != "":
            weights = torch.load(config.audio_encoder.pretrained)
            self.audio_encoder.load_state_dict(weights)
        
        if config.ln_audio != "":
            print("Init LN-audio from CKPT")
            weights = torch.load(config.ln_audio)
            self.ln_audio.load_state_dict(weights)

        # Setup Q Former
        self.qformer, self.query_tokens = self.initQformer(
            config.num_query_tokens, config.audio_encoder.embed_dim_audio
        )
        self.qformer.resize_token_embeddings(len(self.tokenizer))

        # Check if a Checkpoint is provided
        if config.qFormer_ckpt != "":
            # Load qFormer from checkpoint
            weights = torch.load(config.qFormer_ckpt)
            self.qformer.load_state_dict(weights)

        if config.qTokens != "":
            self.query_tokens = torch.load(config.qTokens)

        # TODO explanation 
        self.qformer.cls = None
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Setup Language Model
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(config.LLM.t5_model)
        t5_config = T5Config.from_pretrained(config.LLM.t5_model)
        t5_config.dense_act_fn = "gelu"
        t5_config.repetition_penalty = config.LLM.repetition_penalty
        if config.atRandom:
            print("Caution, init LLM at random, if you init from a checkpoint you can ignore this message")
            self.t5_model = T5ForConditionalGeneration(config=t5_config)
        else:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                config.LLM.t5_model, config=t5_config
            )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            #param.data = param.data.bfloat16() # TODO cast datatype to bfloat16 to achieve larger batch size

        self.t5_proj = nn.Linear(
            self.qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = config.max_txt_len
        self.prompt = config.prompt

        self._apply_lemmatizer = config.apply_lemmatizer
        self._lemmatizer = None

        self.validation_step_outputs = []

    @classmethod
    def from_json(cls, modelConfig, trainConfig):
        with open(trainConfig, "r") as f:
            trainConfigData = json.load(f)
        config = BLAP2_Stage2_Config.from_file(modelConfig)
        return cls(config, trainConfigData)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, modelConfig, trainConfig=""):
        # Load configurations from files
        config = BLAP2_Stage2_Config.from_file(modelConfig)
        if trainConfig != "":
            with open(trainConfig, "r") as f:
                train_Config = json.load(f)
        else:
            print("Caution No training information provided, for inference you can ignore this message")
            train_Config = {}
        # Load the model from the checkpoint
        map_location = None

        if not torch.cuda.is_available():
            print("No GPU found, loading on CPU")
            map_location = torch.device('cpu')
        model = cls.load_from_checkpoint(checkpoint_path,map_location=map_location, config=config, trainConfig=train_Config)

        return model
    
    def remove_forbidden_phrases_with_chance(self, captions, forbidden_phrases, chance=0.3):
        # Remove occurrences of the forbidden phrases based on a random chance
        processed_captions = []
        for cap in captions:
            modified_caption = cap
            for phrase in forbidden_phrases:
                while phrase in modified_caption:
                    if torch.rand(1) < chance:
                        modified_caption = modified_caption.replace(phrase, "", 1)
                    else:
                        # Break the loop if the condition is not met to avoid infinite loop
                        break
            processed_captions.append(modified_caption.strip())
        return processed_captions

    def forward(self, audios, captions, device=None, qualitative_testing=True):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        captions = self.remove_forbidden_phrases_with_chance(captions, self.forbidden_words, self.forbidden_drop)


        # Compute Audio Emebedding via Audio Encoder
        audio_embeds = self.ln_audio(F.avg_pool2d( self.audio_encoder(audios)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        # Obtain output of Q-Former 
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )

        # Map Q-Former output to T5 representation
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        # Generate Prompt tokens
        input_tokens = self.t5_tokenizer(
                [self.prompt] * audio_embeds.size()[0],
                # captions,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
        ).to(device)

        # Generate Target Tokens
        output_tokens = self.t5_tokenizer(
                captions,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
        
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        # Compute Final Token input embedding
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        # Compute Output including loss
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        # ****************************************************** #
        # generate text predictions for small subset of the data #
        # ****************************************************** #
        if qualitative_testing:
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds[:3],
                attention_mask=encoder_atts[:3],
                do_sample=False,
                num_beams=10,
                max_new_tokens=35,
                min_length=20,
                length_penalty=0.5,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            if self._apply_lemmatizer:
                output_text = self._lemmatize(output_text)

            print(output_text)
        return {"loss": loss}
    
    def predict_answers(
        self,
        audios,
        prompt,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        
        length_penalty=-1,
        device=None
    ):
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_embeds = self.ln_audio(F.avg_pool2d( self.audio_encoder(audios)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        input_tokens = self.t5_tokenizer(
            [prompt] * audio_embeds.size()[0], padding="longest", return_tensors="pt"
        ).to(device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=False,
            num_beams=num_beams,
            max_new_tokens=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text


    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    
    #------------------------------------------------#
    #----------------PyTorch Lightning---------------#
    #------------------------------------------------#

    def setup(self, stage=None) -> None:
        trainSetConfig = self.trainConfig["trainSet"]
        set_type = trainSetConfig["type"]
        dataset = ShutterStock(trainSetConfig['data'], trainSetConfig['music']) if set_type == 'ShutterStock' else MusicCaps(trainSetConfig['data'], trainSetConfig['music'])
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        if set_type == 'ShutterStock':
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        else:
            self.train_dataset, self.val_dataset = dataset.createSplit()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.trainConfig["learn_rate"])

    def training_step(self, batch, batch_idx):
        audio, caption = batch
        loss = self(audio, list(caption), qualitative_testing=False)

        # Logging
        self.log('loss_train', loss["loss"].item(), sync_dist=True)
       
        return loss

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return  DataLoader(
            self.train_dataset, 
            batch_size=self.trainConfig["batch_size"], 
            sampler=sampler, 
            drop_last=True
        )

    def val_dataloader(self):

        sampler = DistributedSampler(self.val_dataset)

        return DataLoader(
            self.val_dataset,
            batch_size=self.trainConfig["batch_size"],
            sampler= sampler,
            drop_last=True
        )

    def validation_step(self, batch, batch_idx):
        audio, caption = batch
        loss = self(audio, list(caption))
        
        
        self.validation_step_outputs.append(loss["loss"].item())

    def on_validation_epoch_end(self):
        val_loss = np.array(self.validation_step_outputs).mean()
       
       
        self.log('val_loss', val_loss, sync_dist=True)
        self.validation_step_outputs = []
        return {
            'val_loss', val_loss
        }
    