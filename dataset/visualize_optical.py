from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_pair(healthy740,healthy850,hem740,hem850,mask,out_path):
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    d740=hem740-healthy740; d850=hem850-healthy850
    fig,axes=plt.subplots(3,4,figsize=(16,11))
    items=[(healthy740,"Healthy 740"),(hem740,"Hemorrhage 740"),(d740,"Delta 740"),
           (d740/(healthy740+np.finfo(float).tiny),"Relative delta 740"),
           (healthy850,"Healthy 850"),(hem850,"Hemorrhage 850"),(d850,"Delta 850"),
           (d850/(healthy850+np.finfo(float).tiny),"Relative delta 850")]
    for ax,(arr,title) in zip(axes[:2].ravel(),items):
        im=ax.imshow(arr,aspect='auto'); ax.set_title(title); ax.set_xlabel('Detector'); ax.set_ylabel('Source'); fig.colorbar(im,ax=ax,shrink=.75)
    inds=np.argwhere(mask>0)
    if len(inds):
        c=np.rint(inds.mean(axis=0)).astype(int); slices=[mask[:,:,c[2]],mask[:,c[1],:],mask[c[0],:,:]]
        titles=["Mask XY","Mask XZ","Mask YZ"]
        for ax,a,t in zip(axes[2,:3],slices,titles): ax.imshow(a.T,origin='lower',aspect='auto'); ax.set_title(t)
    axes[2,3].axis('off')
    fig.tight_layout(); fig.savefig(out_path,dpi=150); plt.close(fig)
