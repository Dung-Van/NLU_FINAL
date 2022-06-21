from src.trainer import Trainer
from src.data_loader import Dataset
from src.model import LSTM, Design
from src import utils
import numpy as np
import json

cfg = utils.load_cfg("config.cfg")
print(cfg)
trainset = Dataset(data_path=cfg["train"], mode="train")
validset = Dataset(data_path=cfg["valid"], mode="valid", ref_dict=trainset.word_dict, symbol=trainset.symbol)
testset  = Dataset(data_path=cfg["test"] , mode="test" , ref_dict=trainset.word_dict, symbol=trainset.symbol)
testset.set_step(40)

model = LSTM(num_token=len(trainset.word_set), num_inp=cfg["num_inp"], num_hid=cfg["num_hid"], num_layers=cfg["num_lstm"],is_tie=cfg["tie_weight"])

trainer = Trainer(model=model, cfg=cfg)

log = [cfg]
best_ppl = 9999
for epoch in range(cfg["num_epochs"]):
    step = np.random.randint(100,200)
    trainset.set_step(step)
    validset.set_step(step)
    loss_train, ppl_train = trainer.epoch(trainset, mode="train")
    loss_valid, ppl_valid = trainer.epoch(validset, mode="valid")
    loss_test, ppl_test = trainer.epoch(testset, mode="test")
    dict_out = {"eps":epoch,
        "train":{"loss":loss_train,"ppl":ppl_train},
        "valid":{"loss":loss_valid,"ppl":ppl_valid},
        "test":ppl_test}
    log.append(dict_out)
    print(dict_out)

    # if epoch%5:
    #     trainer.half_lr()

    if epoch%5 == 0:
        json.dump(log, open("log.json","w"))
    if ppl_valid < best_ppl:
        print("UPDATE BEST", ppl_valid, best_ppl)
        best_ppl = ppl_valid
        # trainer.save("result/eps_%d.pth"%epoch)
