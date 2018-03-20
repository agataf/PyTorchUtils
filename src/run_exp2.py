#!/usr/bin/env python
__doc__ = """

Training Script

Nicholas Turner, 2017
"""

import os, imp
import collections

import torch

import utils
import train
import loss


def main(**args):

    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)

    utils.set_gpus(params["gpus"])

    utils.make_required_dirs(**params)

    utils.log_tagged_modules(params["modules_used"],
                             params["log_dir"], "train",
                             chkpt_num=params["chkpt_num"])

    start_training(**params)


def fill_params(expt_name, chkpt_num, batch_sz, lr, gpus,
                sampler_fname, model_fname, erode, **args):

    params = {}

    #Model params
    params["in_dim"]       = 1
    params["output_spec"]  = collections.OrderedDict(mit1_label=1, mit2_label=2)
    params["depth"]        = 4
    params["batch_norm"]   = True

    #Training procedure params
    params["max_iter"]    = 1000000
    params["lr"]          = lr
    params["test_intv"]   = 1000
    params["test_iter"]   = 100
    params["avgs_intv"]   = 50
    params["chkpt_intv"]  = 10000
    params["warm_up"]     = 50
    params["chkpt_num"]   = chkpt_num
    params["batch_size"]  = batch_sz

    #Sampling params
    params["data_dir"]     = os.path.expanduser("~/seungmount/research/agataf/datasets/pinky_all")
    assert os.path.isdir(params["data_dir"]),"nonexistent data directory"
    train_vol_list = ["vol19-34_train",  "vol101_train", "vol102_train", "vol103_train", "vol104_train", "vol401_train", "vol501_train", "vol502_train", "vol503_train"]
    val_vol_list = ["vol19-34_val", "vol401_val", "vol501_val", "vol502_val", "vol503_val"]
    
    params["train_sets"]   = [el for el in train_vol_list]
    #params["train_sets1"]   = [el+"_1eroded" for el in train_vol_list]
    #params["train_sets2"]   = [el+"_2eroded" for el in train_vol_list]
    params["val_sets"]   = [el for el in val_vol_list]
    #params["val_sets1"]   = [el+"_1eroded" for el in val_vol_list]
    #params["val_sets2"]   = [el+"_2eroded" for el in val_vol_list]

    #GPUS
    params["gpus"] = gpus

    #IO/Record params
    params["expt_name"]  = expt_name
    params["expt_dir"]   = "experiments/{}".format(expt_name)
    params["model_dir"]  = os.path.join(params["expt_dir"], "models")
    params["log_dir"]    = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"]    = os.path.join(params["expt_dir"], "forward")

    #Use-specific Module imports
    params["sampler_class"] = imp.load_source("Sampler",sampler_fname).Sampler
    params["model_class"]    = imp.load_source("Model",model_fname).Model
    
    #"Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"]     = [ params["in_dim"], params["output_spec"],
                                 params["depth"] ]
    params["model_kwargs"]   = { "bn" : params["batch_norm"] }

    #modules used for record-keeping
    params["modules_used"] = [__file__, model_fname, "layers.py",
                              sampler_fname, "loss.py"]

    return params


def start_training(model_class, model_args, model_kwargs, chkpt_num,
                   lr, train_sets, val_sets, data_dir, **params):

    #PyTorch Model
    net = utils.create_network(model_class, model_args, model_kwargs)
    monitor = utils.LearningMonitor()

    #Loading model checkpoint (if applicable)
    if chkpt_num != 0:
        utils.load_chkpt(net, monitor, chkpt_num,
                         params["model_dir"],
                         params["log_dir"])

    #DataProvider Sampler
    Sampler = params["sampler_class"]
    train_sampler = utils.AsyncSampler(Sampler(data_dir, dsets=train_sets,
                                               mode="train"))

    val_sampler   = utils.AsyncSampler(Sampler(data_dir, dsets=val_sets,
                                               mode="val"))

    loss_fn = loss.BinomialCrossEntropyWithLogits()
    optimizer = torch.optim.Adam( net.parameters(), lr=lr )

    train.train(net, loss_fn, optimizer, train_sampler, val_sampler,
                last_iter=chkpt_num, monitor=monitor, **params)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description= __doc__)

    parser.add_argument("expt_name",
                        help="Experiment Name")
    parser.add_argument("sampler_fname",
                        help="DataProvider Sampler Filename")
    parser.add_argument("model_fname",
                        help="Model Template Filename")
    parser.add_argument("--batch_sz",  type=int, default=1,
                        help="Batch size for each sample")
    parser.add_argument("--chkpt_num", type=int, default=0,
                        help="Checkpoint Number")
    parser.add_argument("--gpus", default=["0"], nargs="+")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--erode", type=bool, default=False)
    args = parser.parse_args()


    main(**vars(args))
