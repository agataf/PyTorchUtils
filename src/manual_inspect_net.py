import h5py
import os
import numpy as np

def write_h5(data,fname):                                                                                                      
  with h5py.File(fname,"w") as f:                          
    f.create_dataset("/main",data=data)    
 
 params = fill_params('exp8', 430000, ['0'], True, 'models/RSUNet_down_interm.py', '', '')
 d=torch.load('experiments/exp8/models/model430000.chkpt') 
 params['model_kwargs']['bn']=False   
 
  net = utils.create_network(**params)  
img = utils.read_h5('/usr/people/agataf/seungmount/research/agataf/datasets/pinky_all/chunk_19585-21632_22657-24704_4003-4258.omni_img.h5') 
img=img[:16,:320,:320]    
img = img.reshape((1,1) + img.shape)  
img = img.astype("float32") / 255. 
v = torch.autograd.Variable(torch.from_numpy(img),volatile=True).cuda() 
out = net(v)
inputconv=net.module.inputconv(net.module.downsample(v))   
convmod0_conv1 = net.module.convmod0.conv1(inputconv)   
convmod0_conv1_bn = net.module.convmod0.bn1(convmod0_conv1)   
convmod0_conv1_act = net.module.convmod0.activation(convmod0_conv1_bn)
ret1 = convmod0_conv1_act.data.cpu().numpy()  
 
write_h5(ret1,"ret1.h5") 

convmod0_conv2 = net.module.convmod0.conv2(convmod0_conv1_act)   
convmod0_conv2_bn = net.module.convmod0.bn2(convmod0_conv2)   
convmod0_conv2_act = net.module.convmod0.activation(convmod0_conv2_bn)
