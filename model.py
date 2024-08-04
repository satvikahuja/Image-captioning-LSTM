import torch
import torch.nn as nn
import torchvision.models as models

#________ENCODER___________
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        #Disable learning prar
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0),-1)
        features = self.embed(features)
        return features

#---------DECODER----------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        #Assigning hidden dimension
        self.hidden_dim = hidden_size
        #Map each word index to a dense word embedding tensor of embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        #creating LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #Initializing linear to apply at last of RNN layer for further prediction
        self.linear = nn.Linear(hidden_size, vocab_size)
        #initializing values for hidden and cell state
        self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size))

    def forward(self, features, captions):
        #remove <end> token form captions and embed captions
        cap_embedding = self.embed(
            captions[:,:-1]
        ) #(bs, cap_length) -> (bs, cap_length-1, embed_size)
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)
        #getting output i.e. score and hidden layer
        #first value: all the hidden states throughout the sequence. second value: the most recent hidden state
        lstm_out, self.hidden = self.lstm(
            embeddings
        ) #(bs, cap_length, hidden_size), (1,bs, hidden_size)
        outputs = self.linear(lstm_out) # (bs, cap_length, vocab_size)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor(inputs) and returns predicted sentence (list of tensor ids of length max_len)
        Args:
            inputs: shape is (1,1,embed_size)
            states: initial hidden state of the LSTM
            max_len: max length of the predicted sentence
        Returns:
            res: list of predicted words indices
        """
        res = []
        # Now we feed the LSTM output and hidden states back into itself to get the caption

        for i in range(max_len):
            lstm_out, states = self.lstm(
                inputs, states
            ) # lstm_out (1,1,hidden_size)
            outputs = self.linear(lstm_out.squeeze(dim=1))
            _, predicted_idx = outputs.max(dim=1) #predicted: (1,1)
            res.append(predicted_idx.item())
            #if the predicted idx is the stop index, the loop stops
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx) # inputs:(1,embed_size)
            #prepare input for next iteration
            inputs = inputs.unsqueeze(1)
        return res







        
        

