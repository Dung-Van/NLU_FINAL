import math
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model
        try:
            self.model.cuda()
        except:
            self.model.to(torch.device("cpu"))
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        self.lr = self.cfg["learning_rate"]
        if self.cfg["optim"] == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=1.2e-6)
        else:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1.2e-6)


    def half_lr(self):
        self.lr = self.lr/2
        for g in self.optim.param_groups:
            g["lr"] = self.lr
            

    def epoch(self, dataset, mode="train", type_="word"):
        if type_ == "word":
            dataloader = DataLoader(dataset, batch_size=self.cfg["batch_size"], num_workers=self.cfg["num_workers"], shuffle=True if mode == "train" else False)
        else:
            assert type_ == "sentences"
            if mode == "test":
                dataloader = DataLoader(dataset, batch_size=self.cfg["batch_size"], num_workers=self.cfg["num_workers"], shuffle=False, collate_fn=dataset.collate_fn)
            else:
                dataloader = DataLoader(dataset, batch_size=self.cfg["batch_size"], num_workers=self.cfg["num_workers"], shuffle=True if mode == "train" else False, collate_fn=dataset.collate_fn)
        epoch_loss = 0

        # Dataloader
        num_test = 0
        for data, target in tqdm(dataloader):
            # Tensor to device
            """
            - Data: (batch_size, sequence_indices)
            - Target: (batch_size, index) -> [indices]
            - Predict: 
            """
            data = data.to(self.model.device())
            target = target.to(self.model.device())

            if mode == "train":
                self.model.train()
                self.optim.zero_grad()

                predict = self.model(data, self.cfg["context_seq"]) #(batch, sequence, hidden_size)
                if self.cfg["context_seq"] is not None:
                    target = target[:,self.cfg["context_seq"]:]
                    # print(target.size(), predict.size())
                target = target.reshape(-1)

                step_loss = self.criterion(predict, target)
                step_loss.backward()
                if self.cfg["clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])
                self.optim.step()
            else:
                self.model.eval()
                with torch.no_grad():
                    predict = self.model(data, None)
                    target = target.reshape(-1)
                    step_loss = self.criterion(predict, target)
            with torch.no_grad():
                valid_taget = [tmp for tmp in target if tmp != -1]
                num_test+= len(valid_taget)

            epoch_loss += step_loss.item()
        avg_loss = epoch_loss / num_test
        return avg_loss, math.exp(avg_loss)


    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return True




            


        
