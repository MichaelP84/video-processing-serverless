import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

from src.rife_trt import rife_trt
# from src.scene_detect import scene_detect

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 8

core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")

## REAL_ESRGAN
def twoX_fast_upscale_inference_clip(video_path, size, clip=None):
    weights = "/root/engines/2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_25k_clamp_op18_onnxslim.onnx"
    # weights = "/root/engines/2x_ModernSpanimationV2_clamp_op20.engine"
    vs_format = vs.RGBS 
        
    clip = core.bs.VideoSource(source=video_path)

    clip = vs.core.resize.Bicubic(clip, format=vs_format, matrix_in_s="709")  # RGBS means fp32, RGBH means fp16
    
    if size == "large":
        clip = core.trt.Model(
            clip,
            engine_path=weights,  
            num_streams=1,
            tilesize=[512, 512],
            overlap=[48, 48],
        )
    else:
        clip = core.trt.Model(
            clip,
            engine_path=weights,  
            num_streams=4,
        )
        
    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")  # you can also use YUV420P10 for example

    return clip

def fourX_fast_upscale_inference_clip(video_path, size, clip=None):
    weights="engines/RealESRGANv2-animevideo-xsx4_fix.engine"
    vs_format = vs.RGBS 
        
    clip = core.bs.VideoSource(source=video_path)

    clip = vs.core.resize.Bicubic(clip, format=vs_format, matrix_in_s="709")  # RGBS means fp32, RGBH means fp16
    if size == "large":
        clip = core.trt.Model(
            clip,
            engine_path=weights,  
            num_streams=1,
            tilesize=[512, 512],
            overlap=[48, 48],
        )
    else:
        clip = core.trt.Model(
            clip,
            engine_path=weights,  
            num_streams=2,
        )
    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")  # you can also use YUV420P10 for example

    return clip


## RIFE, skip duplicate frames
# calculate metrics
def metrics_func(clip):
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    return core.vmaf.Metric(clip, offs1, 2)

def slow_motion_interpolate_inference_clip(video_path, multi):
    original_clip = core.bs.VideoSource(source=video_path)

    clip = core.resize.Bicubic(
        original_clip, format=vs.RGBH, matrix_in_s="709"
    )  # RGBS means fp32, RGBH means fp16

    # interpolation
    clip = rife_trt(
        clip,
        multi=multi,
        scale=1.0,
        device_id=0,
        num_streams=2,
        engine_path="/root/engines/rife.engine",
        # engine_path="/root/engines/rife422_v2_ensembleFalse_op20_fp16_clamp.engine",
        # engine_path="/root/engines/rife415_v2_ensembleFalse_op20_fp16_clamp.engine",
    )
    
    clip = core.std.AssumeFPS(clip, original_clip)

    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip

def interpolate_inference_clip(video_path, multi):
    original_clip = core.bs.VideoSource(source=video_path)

    clip = core.resize.Bicubic(
        original_clip, format=vs.RGBH, matrix_in_s="709"
    )  # RGBS means fp32, RGBH means fp16

    # interpolation
    clip = rife_trt(
        clip,
        multi=multi,
        scale=1.0,
        device_id=0,
        num_streams=2,
        engine_path="/root/engines/rife.engine",
        # engine_path="/root/engines/rife422_v2_ensembleFalse_op20_fp16_clamp.engine",
        # engine_path="/root/engines/rife415_v2_ensembleFalse_op20_fp16_clamp.engine",
    )
    
    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip