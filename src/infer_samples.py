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
