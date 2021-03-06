#!/usr/bin/env python
__doc__ = """

Inference functions

Nicholas Turner <nturner.cs@princeton.edu>, 2017
"""

import time

import torch
import numpy as np

import utils


def forward(net, scanner, scan_spec, activation=None):

    start = time.time()
    while True:
        
        inputs = scanner.pull()

        if inputs is None:
          break

        inputs = make_variables(inputs)

        outputs = run_forward_pass(net, inputs, activation)
        print("in forward, type of outputs", type(outputs))
        print("in forward, output[0]", outputs[0])
        

        outputs = push_outputs(scanner, outputs, scan_spec)
        print(type(outputs))
        print(outputs.keys())
        end = time.time()
        print("Elapsed: %3f" % (end-start))
        start = end

    return outputs


def make_variables(inputs):
    expanded = [ np.expand_dims(arr, axis=0) for (k,arr) in inputs.items() ]
    return [ utils.make_variable(arr, volatile=True) for arr in expanded ]


def run_forward_pass(net, inputs, activation=None):
    
    outputs = net(*inputs)

    if activation is not None:
        outputs = list(map( activation, outputs ))

    return outputs


def push_outputs(scanner, outputs, scan_spec):

    fmt_outputs = {}
    #print("scan spec keys", scan_spec.keys())
    for (i,k) in enumerate(scan_spec.keys()):
        #print("extract data shape", extract_data(outputs[i]).shape)
        fmt_outputs[k] = extract_data(outputs[i])

    return fmt_outputs
    #scanner.push(fmt_outputs)


def extract_data(expanded_variable):
    return np.squeeze( expanded_variable.data.cpu().numpy(), axis=(0,) )
