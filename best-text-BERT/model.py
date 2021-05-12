from abc import ABC
import torch.nn as nn
from pytorch_transformers import BertModel


class TRECCARModel(nn.Module, ABC):

    def __init__(self, config, freeze_bert=False):
        super(TRECCARModel, self).__init__()
        self.bert_layer = BertModel.from_pretrained(config["bert_model_file"],
                                                    cache_dir=config.data['pretrained_download_dir'])
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.cls_layer = nn.Linear(config.fineTunerModel['num_hidden_units_1'],
                                   config.fineTunerModel['num_classes'])

    def forward(self, seq, attn_masks, type_ids):
        """
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contribution of PAD tokens
        """
        cont_reps, cls_head = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=type_ids)
        # cls_rep = cont_reps[:, 0]  # TODO: why selecting index-0 ?
        return self.cls_layer(cls_head)

