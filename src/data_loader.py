import os
import torch
import json

default_symbol = {"start":"<start>", "end":"<end>", "unknown":"<unkn>", "pad":"<pad>"}

class Dataset:
    def __init__(self, data_path, mode="train", symbol=default_symbol, ref_dict=None):
        assert os.path.isfile(data_path), data_path
        assert mode in ["train", "valid", "test"], mode
        self.mode = mode
        self.symbol = symbol

        self.word_set = set()
        self.word_set.update(list(self.symbol.values()))

        self.raw_data   = self.add_data(data_path)
        self.word_dict  = self.update_word_dict(ref_dict)
        self.data_idxes, word_count = self.get_data_idxes(self.raw_data)
        # print("mode")
        json.dump(word_count, open("%s_count.json"%mode,"w"))


    def set_step(self, step):
        self.step = step
        print("sequence step set to:", step)
        

    def add_data(self, data_path):
        data = []
        for line in open(data_path).readlines():
            words = [self.symbol["start"]] + line.strip().split(" ") + [self.symbol["end"]]
            self.word_set.update(words)
            data += words
        return data


    def update_word_dict(self, ref_dict=None):
        if ref_dict is None:
            word_dict = {word: word_idx for word_idx, word in enumerate(self.word_set)}
        else:
            word_dict = ref_dict.copy()
            print("here")
            # Add symbol to list
            for values in self.symbol.values():
                if values not in word_dict:
                    print(values)
                    word_dict[values] = max(list(word_dict.values()))+1

            # Check word_set
            for word in self.word_set:
                if word not in word_dict:
                    print("not_in")
                    word_dict[word] = word_dict[self.symbol["unknown"]]
        return word_dict
    

    def get_data_idxes(self, data):
        out = []
        count = {}
        for word in data:
            if word not in self.word_dict:
                word = self.symbol["unknown"] 
            out.append(self.word_dict[word])
            if word not in count:
                count[word] = 1
            else:
                count[word]+=1
        return out, count


    def __len__(self):
        return int((len(self.data_idxes)-1)/self.step)


    def __getitem__(self, index):
        data   = self.data_idxes[index*self.step    : index*self.step + self.step]
        target = self.data_idxes[index*self.step + 1: index*self.step + self.step + 1]
        return torch.tensor([data], dtype=torch.int).view(-1), torch.tensor(target, dtype=torch.long).view(-1)


class Dataset_Sentences(Dataset):
    def __init__(self, data_path, mode="train", symbol=default_symbol, ref_dict=None):
        assert os.path.isfile(data_path), data_path
        assert mode in ["train", "valid", "test"], mode
        super(Dataset_Sentences, self).__init__(data_path, mode=mode, symbol=symbol, ref_dict=ref_dict)

        self.data = []
        tmp_sentences = []
        for data in self.data_idxes:
            tmp_sentences.append(data)
            if data == self.word_dict[self.symbol["end"]]:
                self.data.append(tmp_sentences)
                tmp_sentences = []
        # self.set_step(3)


    def set_step(self, step):
        self.step = step
        print("Num sentences set to:", step)

    def __len__(self):
        # if self.mode != "test":
        #     return len(self.data)
        # else:
        #     return int(len(self.data)/self.step)
        return int(len(self.data)/self.step)
        


    def __getitem__(self, index):
        # if self.mode == "test":
        #     index = index*self.step
        index = index*self.step
        
        # data => (sentence_idx, word_sequence)
        data = self.data[index:index + self.step]
        if len(data) < self.step:
            max_len = 0
        else:
            max_len = max([len(sentence) for sentence in data])
        return data, max_len

    
    def collate_fn(self, batch):
        """
        batch: list (batch, (data, max_len))
        data:    [batch, sentences, words]
        target:  [batch, sentences, words] -> [batch * sentences, words]
        """
        data, target = [], []

        batch_max_len = max([max_len[1] for max_len in batch])
        for tmp_batch in batch:
            if tmp_batch[1] == 0: continue
            tmp_sent, tmp_target = [], []
            for tmp_raw in tmp_batch[0]:
                num_pad = batch_max_len - len(tmp_raw)
                assert num_pad >= 0
                tmp1 = (tmp_raw + [self.word_dict[self.symbol["pad"]]]*num_pad)[:-1]
                tmp2 = tmp_raw[1:] + [-1]*num_pad
                assert len(tmp1) ==  batch_max_len -1, (len(tmp1), batch_max_len)
                assert len(tmp2) ==  batch_max_len -1, (len(tmp2), batch_max_len)
                assert len(tmp1) == len(tmp2), (len(tmp1), len(tmp2))
                tmp_sent.append(tmp1)
                tmp_target.append(tmp2) #ignore idx cross_entropy
            data.append(tmp_sent)
            target.append(tmp_target)
            assert len(data) == len(target)
        assert len(data) == len(target)
        
        try:
            data = torch.tensor(data)
            target = torch.tensor(target)
        except Exception as e:
            import json
            json.dump([data, target], open("test.json","w"))
            print(e)
            for batch_idx in range(len(data)):
                print(len(data[batch_idx]), len(target[batch_idx]))
                for seq_idx in range(len(data[batch_idx])):
                    print(len(data[batch_idx][seq_idx]), len(target[batch_idx][seq_idx]))
        try:
            target = target.view(-1, batch_max_len-1)
        except:
            print(target.size(), batch_max_len-1)

        return data, target

if __name__ == "__main__":
    X = Dataset_Sentences("data/ptb.train.txt")
