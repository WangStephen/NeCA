import torch
import torch.nn as nn

def run_network(inputs, fn, netchunk):
    """
    Prepares inputs and applies network "fn".
    """
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    out_flat = torch.cat([fn(uvt_flat[i:i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)], 0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    return out 