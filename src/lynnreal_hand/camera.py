"""Convert physical camera poses using the trained H3 camera contract."""
from pathlib import Path
import numpy as np
from lynnreal_core.backends.minimax_h3.camera import first_frame_relative_w2c, H3_FIXED_FX, H3_FIXED_FY
from lynnreal_core.backends.minimax_h3.geometry import latent_aligned_pixel_indices


def convert_camera(source,output,*,frames):
    values=np.load(source,allow_pickle=False)
    poses=values['camera_c2w'] if 'camera_c2w' in values else values['pose']
    if poses.shape!=(frames,4,4):raise ValueError('Raw absolute C2W must contain one pose per genuine video frame')
    indices=latent_aligned_pixel_indices(frames,allow_partial=True)
    views=first_frame_relative_w2c(poses.astype(np.float32))[indices]
    K=np.repeat(np.eye(3,dtype=np.float32)[None],len(indices),axis=0)
    K[:,0,0]=H3_FIXED_FX;K[:,1,1]=H3_FIXED_FY;K[:,0,2]=K[:,1,2]=.5
    # Logd4 is applied by attention, not here. Do not invert or transform again.
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,camera_viewmats=views,camera_K=K,source_frame_indices=indices)
    return output
