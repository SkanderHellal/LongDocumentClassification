import torch
from torch import nn
from transformers import (
    CamembertPreTrainedModel,
    CamembertModel,
    CamembertClassificationHead
)   

class CamembertForLongSequenceClassification(CamembertPreTrainedModel):
    def __init__(config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.aggregation_method = config.aggregation_method

        assert self.aggregation_method in [
            "max_pooling", "average_pooling", "attention"
            ]

        self.model = CamembertModel(config, add_pooling_layer=False)    
        self.classifier = CamembertClassificationHead(config)
        
        self.post_init()

        if self.aggregation_method == "attention":
            self.attention = AttentionLayer(
                config.attention_dropout,
                config.hidden_dim,
                config.attention_unit,
                config.attention_hops
            )

    def forward(self, input_ids, attention_mask, nb_sentence_per_doc):
        
        output = self.model(input_ids, attention_mask)

        # max pooling aggregation
        if self.aggregation_method == "max_pooling":
            aggregated_output = torch.stack(
                [torch.max(output[nb,:,:], dim=0) for nb in nb_sentence_per_doc]
            )
        # mean pooling aggregation    
        elif self.aggregation_method == "mean_pooling":
            aggregated_output = torch.stack(
                [torch.mean(output[nb,:,:], dim=0) for nb in nb_sentence_per_doc]
            )
        # attention method aggregation    
        else:
            aggregated_output = torch.stack(
                [self.attention(output[nb,:, :]) for nb in nb_sentence_per_doc]
            )

        return aggregated_output            
                


class AttentionLayer(nn.Module):
    def __init__(
        self,dropout, n_hidden, attention_unit, attention_hops
    ):

        self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(n_hidden, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.tanh = nn.Tanh()
        self.attention_hops = attention_hops

        self.init_weight()

    def init_weight(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)    

    def forward(self, input):
        dims = input.size()    
        squeezed_input = input.view(-1, dim[2])

        transformed_input = torch.transpose(input, 0, 1).contiguous()
        transformed_input = transformed_input.view(dims[0], 1, dims[1]) 
        concatenated_input = [transformed_input for i in range(self.attention_hops)]
        concatenated_input = torch.cat(concatenated_input, 1) 

        hidden = self.tanh(self.ws1(self.drop(squeezed_input))) 
        alphas = self.ws2(hidden).view(dims[0], dims[1], -1)  
        alphas = torch.transpose(alphas, 1, 2).contiguous()  
        alphas = nn.Softmax()(alphas.view(-1, dims[1])) 
        alphas = alphas.view(dims[0], self.attention_hops, dims[1])  
        return torch.bmm(alphas, input)