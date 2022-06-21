from src.trainer import Trainer
from src.data_loader import Dataset_Sentences
from src.model import LSTM, Design
from src import utils
import numpy as np
import json

cfg = utils.load_cfg("config.cfg")
print(cfg)
trainset = Dataset_Sentences(data_path=cfg["train"], mode="train")
validset = Dataset_Sentences(data_path=cfg["valid"], mode="valid", ref_dict=trainset.word_dict, symbol=trainset.symbol)
testset  = Dataset_Sentences(data_path=cfg["test"] , mode="test", ref_dict=trainset.word_dict, symbol=trainset.symbol)

model = Design(num_token=len(trainset.word_set), num_inp=cfg["num_inp"], num_hid=cfg["num_hid"], num_layers=cfg["num_lstm"])
testset.set_step(3)
validset.set_step(3)

trainer = Trainer(model=model, cfg=cfg)

log = [cfg]
best_ppl = 9999
for epoch in range(cfg["num_epochs"]):
    trainset.set_step(np.random.randint(1,3))
    loss_train, ppl_train = trainer.epoch(trainset, mode="train", type_="sentences")
    loss_valid, ppl_valid = trainer.epoch(validset, mode="valid", type_="sentences")
    loss_test, ppl_test = trainer.epoch(testset, mode="test", type_="sentences")
    dict_out = {"eps":epoch,
        "train":{"loss":loss_train,"ppl":ppl_train},
        "valid":{"loss":loss_valid,"ppl":ppl_valid},
        "test":ppl_test}
    log.append(dict_out)
    print(dict_out)

    if epoch%5 == 0:
        json.dump(log, open("log.json","w"))
    if ppl_valid < best_ppl:
        print("UPDATE BEST", ppl_valid, best_ppl)
        best_ppl = ppl_valid
        # trainer.save("result/eps_%d.pth"%epoch)
