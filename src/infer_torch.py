#!/usr/bin/env python

import u
import caffe


def main(prototxt_fname, param_fname, *sample_fnames):

    caffe.set_mode_gpu()
    caffe.set_device(0)
    print("about to read in net")
    net = caffe.Net(prototxt_fname, param_fname, caffe.TRAIN)
    print("read in net")
    samples = read_samples(sample_fnames)
    print("read samples")

    for (i,sample) in enumerate(samples):
        output, keys = infer_sample(net,sample)
        print("output", i, "inferred")
        write_output(output, keys)


def read_samples(fnames):
    return [(u.read_file(fname)).astype("float32") / 255. for fname in fnames]


def infer_sample(net, sample):
    print("in infer")
    if len(sample.shape) == 3:
        sample = sample.reshape((1,)+sample.shape)

    net.blobs["input"].data[0,...] = sample[:,:16,:320, :320]
    print("about to run forward")
    pre_output = net.forward()["psd_label"][0,0,...]
    output = net.forward()["upsampl"][0,0,...]
    print net.blobs.keys()
    els = []
    for el in net.blobs.keys():
        els.append(net.blobs[el].data[0,3:7,...])
    
    
#     top = net.blobs["sum0_d3"].data[0,...]
#     ups2 = net.blobs["upsample_d2"].data[0,...]
#     rszconv2 = net.blobs["merge_d2"].data[0,...]

    return els, net.blobs.keys()


def write_output(output, keys):
    for i, key in enumerate(keys):
        u.write_file(output[i], "output/"+str(key)+".h5")
#     u.write_file(output[0], "pre_output")
#     u.write_file(output[1], "output_{num}.h5".format(num=num))
#     u.write_file(output[2], "top1_{num}.h5".format(num=num))
#     u.write_file(output[3], "top2_{num}.h5".format(num=num))
#     u.write_file(output[4], "top3_{num}.h5".format(num=num))
#     u.write_file(output[5], "top4_{num}.h5".format(num=num))
#     u.write_file(output[6], "input_{num}.h5".format(num=num))
#     u.write_file(output[7], "downsampl_{num}.h5".format(num=num))
#     u.write_file(output[8], "convi_{num}.h5".format(num=num))
#     u.write_file(output[9], "conv1_d0_{num}.h5".format(num=num))
#     u.write_file(output[10], "conv2_d0_{num}.h5".format(num=num))
#     u.write_file(output[11], "conv2_d0_elu2_d0_0_split_0_{num}.h5".format(num=num))
#     u.write_file(output[12], "conv2_d0_elu2_d0_0_split_1_{num}.h5".format(num=num))
#     u.write_file(output[13], "pool_d1_{num}.h5".format(num=num))  
#     u.write_file(output[14], "conv0_d1_{num}.h5".format(num=num))  
#     #     u.write_file(output[2], "ups2_{num}.h5".format(num=num))
# #     u.write_file(output[3], "rszconv2_{num}.h5".format(num=num))

    #for (k,v) in output_dict.items():
    #    u.write_file(v[0,0,...], "{k}_{num}.h5".format(k=k,num=num))


if __name__ == "__main__":
    from sys import argv

    main(*argv[1:])
agataf@seungworkstation1003:~/seungmount/research/agataf/mitotools/docker$ ls
deploy.prototxt    forward.pyc       input_0.h5  model430000.chkpt  RSU_caffe.py          run_local.sh    u.py
exp8_chkpt430k.h5  infer_samples.py  layers.py   output             RSUNet_downsample.py  run.sh          u.pyc
forward.py         infer_torch.py    layers.pyc  package_RSUNet.py  run_docker.sh         train.prototxt
agataf@seungworkstation1003:~/seungmount/research/agataf/mitotools/docker$ cat infer_torch.py 
#!/usr/bin/env python

import os, imp
import collections

import torch
from torch.nn import functional as F
import dataprovider as dp

import forward
import utils


def main(noeval, **args):

    #args should be the info you need to specify the params
    # for a given experiment, but only params should be used below
    params = fill_params(**args)

    utils.set_gpus(params["gpus"])

    net = utils.create_network(**params)
    if not noeval:
        net.eval()

    utils.log_tagged_modules(params["modules_used"], params["log_dir"],
                             params["log_tag"], params["chkpt_num"])

    for dset in params["dsets"]:
        print(dset)

        fs = make_forward_scanner(dset, **params)

        output = forward.forward(net, fs, params["scan_spec"],
                                 activation=params["activation"])

        save_output(output, dset, **params)


def fill_params(expt_name, chkpt_num, gpus,
                nobn, model_fname, dset_names, tag):

    params = {}

    #Model params
    params["in_dim"]      = 1
    params["output_spec"] = collections.OrderedDict(psd_label=1)
    params["depth"]       = 4
    params["batch_norm"]  = not(nobn)
    params["activation"]  = F.sigmoid
    params["chkpt_num"]   = chkpt_num

    #GPUS
    params["gpus"] = gpus

    #IO/Record params
    params["expt_name"]   = expt_name
    params["expt_dir"]    = "experiments/{}".format(expt_name)
    params["model_dir"]   = os.path.join(params["expt_dir"], "models")
    params["log_dir"]     = os.path.join(params["expt_dir"], "logs")
    params["fwd_dir"]     = os.path.join(params["expt_dir"], "forward")
    params["log_tag"]     = "fwd_" + tag if len(tag) > 0 else "fwd"
    params["output_tag"]  = tag

    #Dataset params
    params["data_dir"]    = os.path.expanduser(
                            "~/seungmount/research/agataf/datasets/pinky_all")
    assert os.path.isdir(params["data_dir"]),"nonexistent data directory"
    params["dsets"]       = dset_names
    params["input_spec"]  = collections.OrderedDict(input=(16,320,320)) #dp dataset spec
    params["scan_spec"]   = collections.OrderedDict(psd=(1,16,320,320))
    params["scan_params"] = dict(stride=(0.5,0.5,0.5), blend="bump")

    #Use-specific Module imports
    params["model_class"]  = imp.load_source("Model", model_fname).Model

    #"Schema" for turning the parameters above into arguments
    # for the model class
    params["model_args"]   = [params["in_dim"], params["output_spec"],
                             params["depth"] ]
    params["model_kwargs"] = { "bn" : params["batch_norm"] }

    #Modules used for record-keeping
    params["modules_used"] = [model_fname, "layers.py"]

    return params


def make_forward_scanner(dset_name, data_dir, input_spec,
                         scan_spec, scan_params, **params):
    """ Creates a DataProvider ForwardScanner from a dset name """

    # Reading EM image
#    img = utils.read_h5(dset_name)
    print("image path", os.path.join(data_dir, dset_name + "_img.h5"))
    img = utils.read_h5(os.path.join(data_dir, dset_name + "_img.h5"))[:16,:320,:320]
    print("image dimensions", img.shape)
    img = (img / 255.).astype("float32")

    # Creating DataProvider Dataset
    vd = dp.VolumeDataset()

    vd.add_raw_data(key="input", data=img)
    vd.set_spec(input_spec)

    # Returning DataProvider ForwardScanner
    return dp.ForwardScanner(vd, scan_spec, params=scan_params)


def save_output(output, dset_name, chkpt_num, fwd_dir, output_tag, **params):
    """ Saves the volumes within a DataProvider ForwardScanner """

    for k in output.outputs.data.iterkeys():
        print(type(output.outputs))
        print(output.outputs.keys())
        output_data = output.outputs.get_data(k)

        if len(output_tag) == 0:
            basename = "{}_{}_{}.h5".format(dset_name, k, chkpt_num)
        else:
            basename = "{}_{}_{}_{}.h5".format(dset_name, k, 
                                               chkpt_num, output_tag)

        full_fname = os.path.join( fwd_dir, basename )

        utils.write_h5(output_data[0,:,:,:], full_fname)


#============================================================



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("expt_name",
                        help="Experiment Name")
    parser.add_argument("model_fname",
                        help="Model Template Filename")
    parser.add_argument("chkpt_num", type=int,
                        help="Checkpoint Number")
    parser.add_argument("dset_names", nargs="+",
                        help="Inference Dataset Names")
    parser.add_argument("--nobn", action="store_true",
                        help="Whether net uses batch normalization")
    parser.add_argument("--gpus", default=["0"], nargs="+")
    parser.add_argument("--noeval", action="store_true",
                        help="Whether to use eval version of network")
    parser.add_argument("--tag", default="",
                        help="Output (and Log) Filename Tag")


    args = parser.parse_args()

    main(**vars(args))
