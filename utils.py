#Extract subvolumes from big volumes
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
from numpy import linspace, zeros, array
import nibabel
from math import log10, sqrt

def extractSubDirections(in_vol, inds,b=False:
    out_vol = []
    for i in inds:
        if not b:
            out_vol.append(in_vol[:, :, :, i])
        else:
            out_vol.append(in_vol[i])
    np.array(out_vol)
    if not b:
        out_vol = np.swapaxes(np.swapaxes(np.swapaxes(out_vol,0,1),1,2),2,3)
    return np.asarray(out_vol)

def mergeStacks(stacks):
    out_vol = []
    for i,stack in enumerate(stacks):
        for j in range(stack.shape[-1]):
            out_vol.append(stack[:,:,:,j])
    out_vol = np.asarray(out_vol)
    out_vol = np.swapaxes(np.swapaxes(np.swapaxes(out_vol, 0, 1), 1, 2), 2, 3)
    return np.asarray(out_vol)

def mergeBvals(b):
    out = []
    for i,stack in enumerate(b):
        for j in range(b[i].shape[-1]):
            out.append(b[i][j])
    return np.array(out)

def mergeBvecs(b):
    out = []
    for i,stack in enumerate(b):
        for j in range(b[i].shape[0]):
            out.append(b[i][j][:])
    return np.array(out)


def computeSNR(in_vol,n_len_noise,hw_sig,offset=5):
    #Signal computed on threshold and noise on 4 corners of the phantom
    b0 = in_vol[:,:,:,0]
    midx = round(b0.shape[0]/2)
    midy = round(b0.shape[1]/2)
    midz = round(b0.shape[2]/2)

    b0_sig = b0[midx-hw_sig[0]:midx+hw_sig[0],midy-hw_sig[1]:midy+hw_sig[1],midz-hw_sig[2]:midz+hw_sig[2]]
    b0_noise = b0[offset:offset+n_len_noise,offset+n_len_noise,:]+b0[-n_len_noise-offset:-offset,offset:offset+n_len_noise,:]+ b0[offset:offset+n_len_noise,-n_len_noise-offset:-offset,:]+b0[-n_len_noise-offset:-offset,-n_len_noise-offset:-offset,:]

    b0_snr = np.mean(b0_sig)/np.std(b0_noise)

    b = in_vol[:,:,:,1:]
    b_sig = b[midx-hw_sig[0]:midx+hw_sig[0],midy-hw_sig[1]:midy+hw_sig[1],midz-hw_sig[2]:midz+hw_sig[2]]
    b_noise = b[offset:offset + n_len_noise, offset + n_len_noise, :] + b[-n_len_noise - offset:-offset, offset:offset + n_len_noise, :] + b[offset:offset + n_len_noise, -n_len_noise - offset:-offset, :] + b[-n_len_noise - offset:-offset,-n_len_noise - offset:-offset,:]

    b_snr = np.mean(b_sig)/np.std(b_noise)
    return b0_snr,b_snr
