#!/usr/bin/env python3
__doc__= """

Sampling class for synapse training

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""
import os
import utils
import dataprovider as dp
import h5py
from scipy import misc

def read_file(fname):

    f = h5py.File(fname)
    d = f["/main"].value
    f.close()

    return d


class Sampler(object):

    def __init__(self, datadir, dsets=[], mode="train", erode=0, patchsz=(16,160,160)):

      assert mode in ["train","val","test"]

      datadir = os.path.expanduser(datadir)

      volnames = ["input","psd_label"]
      spec = { name : patchsz for name in volnames }

      self.dp = self.build_data_provider(datadir, spec, mode, dsets, erode)


    def __call__(self, **kwargs):
      return self.dp("random", **kwargs)


    def build_data_provider(self, datadir, spec, mode, dsets, erode):

      vdp = dp.VolumeDataProvider()

      for dset in dsets:
        vdp.add_dataset( self.build_dataset(datadir, spec, dset, erode) )
      if mode == "train":
          vdp.set_sampling_weights([20, 4, 1, 4, 4, 1, 1, 1, 4])
      if mode == "val":
          vdp.set_sampling_weights([1,1,1,1,1])
      vdp.set_augmentor(self._aug(mode))
      vdp.set_postprocessor(self._post())

      return vdp


    def build_dataset(self, datadir, spec, dset_name, erode):
      
      print(dset_name)
      img = read_file(os.path.join(datadir, dset_name + "_img.h5"))
      mit_str = "_mit.h5"
      if erode:
        mit_str = "_"+str(erode)+"erode"+"_mit.h5"
      mit = read_file(os.path.join(datadir, dset_name + mit_str)).astype("float32")
#       print(img.shape)
#       print(mit.shape)
#      seg = read_file(os.path.join(datadir, dset_name + "_seg.h5"))

      img = dp.transform.divideby(img, val=255.0, dtype="float32")
    
      mit[mit != 0] = 1 #Binarizing psds
      vd = dp.VolumeDataset()
      vd.add_raw_data(key="input",      data=img)
      vd.add_raw_data(key="psd_label",  data=mit)
      # vd.add_raw_data(key="psd_mask", data=msk)

      vd.set_spec(spec)
      return vd


    def _aug(self, mode):

      aug = dp.Augmentor()
      if mode == "train":
        aug.append('misalign', max_trans=17.0)
      aug.append('missing', max_sec=5, mode='mix',random_color=True)
      aug.append('blur', max_sec=5, mode='mix')
      if mode == "train":
        aug.append('warp')
        aug.append('greyscale', mode='mix')
        aug.append('flip')
      return aug


    def _post(self):
      post = dp.Transformer()
      return post
