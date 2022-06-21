from multiprocessing import context
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_token, num_inp, num_hid, num_layers=1, is_tie=False):
        super(LSTM, self).__init__()
        self.dummy = nn.Parameter(torch.empty(0))
        # Embedding
        self.dropi = nn.Dropout(0.25)
        self.droph = nn.Dropout(0.5)
        self.drope = nn.Dropout(0.4)

        self.embedding = nn.Embedding(num_token, num_inp)
        if is_tie:
            print("Using tie weight -> using num_inp instead of num_hid")
            self.lstm = nn.LSTM(num_inp, num_inp, num_layers)
            self.out = nn.Linear(num_inp, num_token)
            self.out.weight = self.embedding.weight
            self.num_hid = num_inp
        else:
            self.num_hid = num_hid
            self.lstm = nn.LSTM(num_inp, num_hid, num_layers)
            self.out = nn.Linear(num_hid, num_token)

    def device(self):
        return self.dummy.device

    def reset(self):
        self.is_begin = True

    def set(self):
        self.is_begin = False

    def forward(self, input, context_seq=40):
        """
        input:          (batch_size, sequence)
        word_embbeded:  (batch_size, sequence, embbeding_size) -> (sequence, batch_size, embbeding_size)
        output/hiddens: (sequence, batch_size, num_hidden)
        """
        self.reset()

        word_embbeded = self.drope(self.embedding(input))
        word_embbeded = word_embbeded.permute(1,0,2)

        # output, (hiddens, cells) = self.lstm(word_embbeded)
        output_tot = []
        for sequence_idx in range(word_embbeded.size(0)):
            word = word_embbeded[sequence_idx].unsqueeze(0)
            if self.is_begin:
                output, (hiddens, cells) = self.lstm(word, None)
            else:
                output, (hiddens, cells) = self.lstm(word, (hiddens, cells))
            # hiddens = self.droph(hiddens)
            output_tot.append(output)
            self.set()

        # output: (sequence, batch, H_out) -> (batch, sequence, hidden)
        output = self.droph(torch.cat(output_tot, dim=0))
        # print(output.size())
        if context_seq is not None: output =  output[context_seq:, :, :]
        output = output.permute(1,0,2).reshape(-1, self.num_hid)
        return self.dropi(self.out(output))




class Design(nn.Module):
    def __init__(self, num_token=10001, num_inp=200, num_hid=700, num_layers=1):
        super(Design, self).__init__()
        self.dummy = nn.Parameter(torch.empty(0))

        # Embedding
        self.embedding = nn.Embedding(num_token, num_inp)

        self.num_hid = num_hid
        self.lstm_sentence = nn.LSTM(num_inp, num_hid, num_layers, dropout=0.25)
        self.lstm_context  = nn.LSTM(num_hid, num_hid, num_layers, dropout=0.25)

        self.out = nn.Linear(num_hid, num_token)
        self.dropo = nn.Dropout(0.25)
        self.drope = nn.Dropout(0.4)


    def device(self):
        return self.dummy.device

    def reset_hidden(self):
        self.is_begin = True

    def set_hidden(self):
        self.is_begin = False


    def forward(self, paragraph, context_seq=None):
        """
        paragraph:      (batch_size, sentences, word_sequence)
        sentence :      (batch_size, word_sequence)
        embbeded :      (batch_size, word_sequence, word_features)
        word     :      (1, batch_size, word_features)
        hidden_s :      (1, batch_size, hidden_size)
        hidden_c :      (1, batch_size, hidden_size)
        context_cells:  (1, batch_size, num_hidden)
        output/hiddens: (1, batch_size, num_hidden)
        """

        # Init hiddens
        self.reset_hidden()

        # Sentences
        out_s = []
        for sentence_idx in range(paragraph.size(1)):
            sentence = paragraph[:,sentence_idx,:]
            embbeded = self.drope(self.embedding(sentence)).permute(1,0,2) # (seq, batch, embbeding)

            if self.is_begin:
                out_, (hidden_s, cells_s) = self.lstm_sentence(embbeded, None)
                _, (hidden_c, context_cells) = self.lstm_context(cells_s, None)
            else:
                out_, (hidden_s, cells_s) = self.lstm_sentence(embbeded, (hidden_s, context_cells))
                _, (hidden_c, context_cells) = self.lstm_context(cells_s, (hidden_c, context_cells))
            out_s.append(out_)
            self.set_hidden()

        # out_s: (sequence, batch, hidden_size) -> (batch, sequence, hidden_size)
        out_s = torch.cat(out_s, dim=0).permute(1,0,2)
        out_s = out_s.reshape(-1, self.num_hid)
        return self.dropo(self.out(out_s))

if __name__ == "__main__":
    paragraph = torch.tensor([[[1,1,2,3,4,5,6],[1,2,2,3,4,5,6]],[[1,1,2,3,4,5,6],[1,2,2,3,4,5,6]]])
    print(paragraph.size())
    x = Design()
    tmp = x(paragraph)
    print(tmp.size())

    # torch.Size([2, 2, 7])
    # torch.Size([1, 2, 1150]) torch.Size([1, 2, 1150]) torch.Size([1, 2, 1150])
    # torch.Size([1, 2, 1150]) torch.Size([1, 2, 1150]) torch.Size([1, 2, 1150])
    # torch.Size([2, 14, 1150])
    # torch.Size([28, 10001])