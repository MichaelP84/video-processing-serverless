import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("/workspace/tensorrt/")
from process_video import interpolate_inference_clip, slow_motion_interpolate_inference_clip, twoX_fast_upscale_inference_clip, fourX_fast_upscale_inference_clip

input_values = args.split(" ")
task = input_values[0]
input_file = input_values[1]

if task == "upscale":
    
    scale = int(input_values[2])
    size = input_values[3]
        
    if scale == 2:
        clip = twoX_fast_upscale_inference_clip(input_file, size)
        clip.set_output()
    
    else:
        clip = fourX_fast_upscale_inference_clip(input_file, size)
        clip.set_output()
            

elif task == "interpolate":
    
    multi = int(input_values[2])
    slowmotion = input_values[3]
    
    if (slowmotion == "True"):
        clip = slow_motion_interpolate_inference_clip(input_file, multi)
        clip.set_output()
        
    else:
        clip = interpolate_inference_clip(input_file, multi)
        clip.set_output()