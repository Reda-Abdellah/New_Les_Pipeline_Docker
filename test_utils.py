import  os, sys, glob,  gc
import nibabel as nii
import numpy as np
import torch
import torch.nn as nn
import statsmodels.api as sm
from scipy.signal import argrelextrema
import numpy as np
import nibabel as nii
#from Registration.registration import ToMNI_ANTS_ref, FLAIR_to_MNI, MASK_to_MNI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seg_majvote_times(FLAIR_t1,FLAIR_t2,MODELS,ps=[64,64,64],
        offset1=32,offset2=32,offset3=32,crop_bg=0):
    MASK = (1-(FLAIR_t1==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)

    out_shape=(FLAIR_t1.shape[0],FLAIR_t1.shape[1],FLAIR_t1.shape[2],2)
    output=np.zeros(out_shape,FLAIR_t1.dtype)
    acu=np.zeros(out_shape[0:3],FLAIR_t1.dtype)

    ii=0 # Network ID
    for model in MODELS:
        for x in range(crop_bg,  FLAIR_t1.shape[0] ,offset1):
            xx = x+ps[0]
            if xx> output.shape[0]:
                xx = output.shape[0]
                x=xx-ps[0]
            for y in range(crop_bg,  FLAIR_t1.shape[1] ,offset2):
                yy = y+ps[1]
                if yy> output.shape[1]:
                    yy = output.shape[1]
                    y=yy-ps[1]
                for z in range(crop_bg,  FLAIR_t1.shape[2] ,offset3):

                    zz = z+ps[2]
                    if zz> output.shape[2]:
                        zz = output.shape[2]
                        z=zz-ps[2]

                    T = np.reshape(   FLAIR_t1[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    F = np.reshape(    FLAIR_t2[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    T=np.concatenate((T,F), axis=4)
                    T=torch.from_numpy(T.transpose((0,4,1,2,3))).float().to(device)
                    lista=np.array([0,1])
                    with torch.no_grad():
                        patches = model(T)
                        patches= patches.cpu().numpy()
                        patches= patches.transpose((0,2,3,4,1))
                    #store result
                    local_patch = np.reshape(patches,(patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4]))
                    output[x:xx,y:yy,z:zz,:]=output[x:xx,y:yy,z:zz,:]+local_patch[0:xx-x,0:yy-y,0:zz-z]
                    ii=ii+1

    SEG= np.argmax(output, axis=3)
    SEG_mask= np.reshape(SEG, SEG.shape[0:3])
    SEG_mask = SEG_mask*MASK
    return SEG_mask

def seg_majvote_times_decoder_with_FMs(FLAIR_t1,FLAIR_t2,MODELS,ps=[64,64,64],
        offset1=32,offset2=32,offset3=32,crop_bg=0,out_dim=2):
    
    seg_filepath= '/Weights/voxel_only_da_v3_k0_nf_24._pt'
    seg_model=torch.load(seg_filepath).to(device).eval()
    
    MASK = (1-(FLAIR_t1==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)

    out_shape=(FLAIR_t1.shape[0],FLAIR_t1.shape[1],FLAIR_t1.shape[2],out_dim)
    output=np.zeros(out_shape,FLAIR_t1.dtype)
    acu=np.zeros(out_shape[0:3],FLAIR_t1.dtype)
    
    for model in MODELS:
        ii=0 # Network ID
        for x in range(crop_bg,  FLAIR_t1.shape[0] ,offset1):
            xx = x+ps[0]
            if xx> output.shape[0]:
                xx = output.shape[0]
                x=xx-ps[0]
            for y in range(crop_bg,  FLAIR_t1.shape[1] ,offset2):
                yy = y+ps[1]
                if yy> output.shape[1]:
                    yy = output.shape[1]
                    y=yy-ps[1]
                for z in range(crop_bg,  FLAIR_t1.shape[2] ,offset3):

                    zz = z+ps[2]
                    if zz> output.shape[2]:
                        zz = output.shape[2]
                        z=zz-ps[2]

                    T = np.reshape(   FLAIR_t1[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    F = np.reshape(    FLAIR_t2[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    T=np.concatenate((T,F), axis=4)
                    T=torch.from_numpy(T.transpose((0,4,1,2,3))).float().to(device)
                    with torch.no_grad():
                        FM41,FM31,FM21,FM11=seg_model.encoder(T[:,0:1,:,:,:])
                        FM42,FM32,FM22,FM12=seg_model.encoder(T[:,1:2,:,:,:])
                        FMs1=[FM41,FM31,FM21,FM11]
                        FMs2=[FM42,FM32,FM22,FM12]
                        patches= model(FMs1,FMs2)
                        patches= patches.cpu().numpy()
                        patches= patches.transpose((0,2,3,4,1))
                    #store result
                    local_patch = np.reshape(patches,(patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4]))
                    output[x:xx,y:yy,z:zz,:]=output[x:xx,y:yy,z:zz,:]+local_patch[0:xx-x,0:yy-y,0:zz-z]
                    ii=ii+1
    SEG= np.argmax(output, axis=3)
    SEG_mask= np.reshape(SEG, SEG.shape[0:3])
    SEG_mask = SEG_mask*MASK
    return SEG_mask

def load_times(FLAIR_1_name,FLAIR_2_name,MASK_name=None):
    FLAIR_img = nii.load(FLAIR_1_name)
    FLAIR_1=FLAIR_img.get_data()
    FLAIR_1=FLAIR_1.astype('float32')
    FLAIR_img = nii.load(FLAIR_2_name)
    FLAIR_2=FLAIR_img.get_data()
    FLAIR_2=FLAIR_2.astype('float32')
    if(not MASK_name==None):
        MASK_img = nii.load(MASK_name)
        MASK = MASK_img.get_data()
        MASK=MASK.astype('int')
        FLAIR_1=FLAIR_1*MASK
        FLAIR_2=FLAIR_2*MASK
    peak = normalize_image(FLAIR_1, 'flair')
    FLAIR_1=FLAIR_1/peak
    peak = normalize_image(FLAIR_2, 'flair')
    FLAIR_2=FLAIR_2/peak
    return FLAIR_1,FLAIR_2

def seg_region(flair, overlap=32):
    ax0= flair.sum(axis=(1,2))
    ax0=np.where(ax0>0)
    #ax0min=ax0[0][0]
    #ax0max=ax0[0][-1]
    ax0min=min(ax0[0])
    ax0max=max(ax0[0])
    ax1= flair.sum(axis=(0,2))
    ax1=np.where(ax1>0)
    #ax1min=ax1[0][0]
    #ax1max=ax1[0][-1]
    ax1min=min(ax1[0])
    ax1max=max(ax1[0])
    ax2= flair.sum(axis=(0,1))
    ax2=np.where(ax2>0)
    #ax2min=ax2[0][0]
    #ax2max=ax2[0][-1]
    ax2min=min(ax2[0])
    ax2max=max(ax2[0])

    if(overlap>0):
        ax0min=max([ax0min-overlap,0])
        ax0max=min([ax0max+overlap, flair.shape[0] ])
        ax1min=max([ax1min-overlap,0])
        ax1max=min([ax1max+overlap, flair.shape[1] ])
        ax2min=max([ax2min-overlap,0])
        ax2max=min([ax2max+overlap, flair.shape[2] ])

    return ax0min,ax0max,ax1min,ax1max,ax2min,ax2max

def get_new_lesions_mni(WEIGHTS, flair1_name, flair2_name, brain_mask_name, strategy='unet'):
    pred_name= flair1_name.replace('nuyl_','').replace('flair','new_lesions')
    MODELS=[]
    for weight in WEIGHTS:
        print(weight)
        MODELS.append(torch.load(weight).to(device).eval())
    FLAIR_1,FLAIR_2 =load_times(flair1_name, flair2_name, brain_mask_name)
    if(strategy=='decoder_with_FMs'):
        SEG_mask=seg_majvote_times_decoder_with_FMs(FLAIR_1, FLAIR_2, MODELS, ps=[64,64,64], offset1=32,offset2=32,offset3=32,crop_bg=0)
    else:
        SEG_mask=seg_majvote_times(FLAIR_1, FLAIR_2, MODELS, ps=[64,64,64], offset1=32,offset2=32,offset3=32,crop_bg=0)
    img = nii.Nifti1Image(SEG_mask.astype(np.uint8), nii.load(flair1_name).affine )
    img.to_filename(pred_name)
    gc.collect() #free memory
    return pred_name

def normalize_image(vol, contrast):
    # copied  MedICL-VU / LesionSeg
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    # print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 0.00

    if contrast.lower() in ["t1", "mprage"]:
        peak = peaks[-1]
    elif contrast.lower() in ['t2', 'pd', 'flair', 'fl']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
    else:
        print("Contrast must be either t1,t2,pd, or flair. You entered %s. Returning 0." % contrast)

    return peak

def get_new_lesions(pred_name,flair1_name,flair2_name,brain_mask_name):
    MODELS=[]
    WEIGHTS= sorted(glob.glob("/anima/WEIGHTS/*.pt"))
    for weight in WEIGHTS:
        print(weight)
        MODELS.append(torch.load(weight).to(device).eval())
    FLAIR_1,FLAIR_2 =load_times(flair1_name, flair2_name, brain_mask_name)
    ax0min,ax0max,ax1min,ax1max,ax2min,ax2max= seg_region(FLAIR_1, overlap=32)
    output=np.zeros(FLAIR_1.shape)
    SEG_mask=seg_majvote_times(FLAIR_1[ax0min:ax0max,ax1min:ax1max,ax2min:ax2max],
                        FLAIR_2[ax0min:ax0max,ax1min:ax1max,ax2min:ax2max],
                        MODELS,ps=[64,64,64],
                        offset1=32,offset2=32,offset3=32,crop_bg=0)
    output[ax0min:ax0max,ax1min:ax1max,ax2min:ax2max]=SEG_mask
    img = nii.Nifti1Image(output.astype(np.uint8), nii.load(flair1_name).affine )
    img.to_filename(pred_name)
    gc.collect() #free memory

def times_to_mni(FLAIR1_name, FLAIR2_name, mask_name,regestred_FLAIR1_name, regestred_FLAIR2_name, reg_mask_name):
    nii_ima = nii.load(filename=FLAIR1_name)
    FLAIR1 = nii_ima.get_fdata()
    affine,header = nii_ima.affine, nii_ima.header
    max_fima = np.max(FLAIR1)
    min_fima = np.min(FLAIR1)
    R = max_fima - min_fima
    if R > 256 or min_fima < 0:
        FLAIR1 = ((FLAIR1 - min_fima) / (max_fima - min_fima)) * 256
        print('saving')
        nii.Nifti1Image(FLAIR1, affine, header).to_filename(FLAIR1_name)

    nii_ima = nii.load(filename=FLAIR2_name)
    FLAIR2 = nii_ima.get_fdata()
    affine,header = nii_ima.affine, nii_ima.header
    max_fima = np.max(FLAIR2)
    min_fima = np.min(FLAIR2)
    R = max_fima - min_fima
    if R > 256 or min_fima < 0:
        FLAIR2 = ((FLAIR2 - min_fima) / (max_fima - min_fima)) * 256
        print('saving')
        nii.Nifti1Image(FLAIR2, affine, header).to_filename(FLAIR2_name)

    # get transformation
    affine_t1_intoMNI = ToMNI_ANTS_ref( FLAIR1_name)
    
    # FLAIR1 to MNI Registration
    FLAIR_to_MNI(affine_t1_intoMNI, FLAIR1_name, regestred_FLAIR1_name)

    # FLAIR2 to MNI Registration
    FLAIR_to_MNI(affine_t1_intoMNI, FLAIR2_name, regestred_FLAIR2_name)

    MASK_to_MNI(affine_t1_intoMNI, mask_name, reg_mask_name)

    return affine_t1_intoMNI