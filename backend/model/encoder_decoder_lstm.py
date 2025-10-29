
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from backend.constants.constant import SOS_token, EOS_token, MAX_LENGTH




class MyModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, dropout_p=0.1):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size

        # Encoder
        self.embedding_enc = nn.Embedding(input_vocab_size, hidden_size)
        self.lstm_enc = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout_enc = nn.Dropout(dropout_p)

        # Attention
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size * 2, hidden_size)  # 2x for bidirectional encoder
        self.Va = nn.Linear(hidden_size, 1)

        # Decoder
        self.embedding_dec = nn.Embedding(output_vocab_size, hidden_size)
        self.lstm_dec = nn.LSTM(hidden_size * 3, hidden_size, batch_first=True)  # embed + 2x context
        self.out = nn.Linear(hidden_size, output_vocab_size)
        self.dropout_dec = nn.Dropout(dropout_p)

        # Projection layers (reduce bi-LSTM encoder outputs)
        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)

    def encode(self, input_vector):
        embedded = self.dropout_enc(self.embedding_enc(input_vector))
        output, (hidden, cell) = self.lstm_enc(embedded)
        return output, (hidden, cell)

    def attention(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

    def forward(self, input_tensor, target_tensor=None):
        # Encode
        encoder_outputs, encoder_hidden = self.encode(input_tensor)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=input_tensor.device)

        # Reduce bi-LSTM states
        h = torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), dim=1)
        c = torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), dim=1)
        decoder_hidden = (self.reduce_h(h).unsqueeze(0), self.reduce_c(c).unsqueeze(0))

        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            embedded = self.dropout_dec(self.embedding_dec(decoder_input))
            query = decoder_hidden[0].permute(1, 0, 2)

            context, attn_weights = self.attention(query, encoder_outputs)
            input_lstm = torch.cat((embedded, context), dim=2)

            output, decoder_hidden = self.lstm_dec(input_lstm, decoder_hidden)
            output = self.out(output)

            decoder_outputs.append(output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions
