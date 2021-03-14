import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        self.hidden = hidden_size
        self.n_layers = num_layers
        self.vocab = vocab_size
        self.char = embed_size
        
        # The architecture of the model
        # Embed the caption
        self.w_embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
 
        self.fc = nn.Linear(hidden_size, vocab_size)
    
        
     
    
    def forward(self, features, captions):
        
        # Remove the last word in the tensor
        captions = captions[:, :-1]
        
        # Embed the captions
        embedded_cap = self.w_embed(captions)
        
        # Concat features abd captions
        input_0 = torch.cat((features.unsqueeze(1), embedded_cap), dim=1)
        
        # Get the features and the captions from the lstm
        x,_= self.lstm(input_0)
        
        # Pass the LSTM output through the linear layer
        x = self.fc(x)
        
        return x
        
       

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        n_list = []
        
        for i in range(20):
            # We pass the inputs through the lstm network
            outputs, hid = self.lstm(inputs, states)
            # Then we pass the outputs of the lstm through the linear layer to get the scores
            all_scores = self.fc(outputs)
            # After that we get the highest score among the socres which will be our next prediced word
            prob, h_score = all_scores.max(2)
            # At the end we append the highest score in our words list
            n_list.append(h_score.item())
            # Then we have to spicify the next input to the last which is the new predicted word
            inputs = self.w_embed(h_score)
            
        return n_list
           
        