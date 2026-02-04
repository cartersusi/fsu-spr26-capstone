"""
image_processing.light_level

Determine the light level of an image 
"""

from libcamera import controls

def night_capture_conf(picam2, image_size, dispaly_size):
    return picam2.create_video_configuration(
        main={"size": image_size, "format": "YUV420"},
        #lores={"size": dispaly_size, "format": "YUV420"} if show_preview else None,
        controls={
            "FrameRate": 30,
            "AeEnable": False,
            "ExposureTime": 1000,
            "AnalogueGain": 1.0,
            "AwbEnable": True,
            "AwbMode": controls.AwbModeEnum.Daylight,
            "Saturation": 1.0,
            "Sharpness": 1.5,
            "Contrast": 1.1
        }
    )

def day_capture_conf(picam2, image_size, dispaly_size):
    return picam2.create_video_configuration(
        main={"size": image_size, "format": "YUV420"},
        #lores={"size": dispaly_size, "format": "YUV420"} if show_preview else None,
        controls={
            "FrameRate": 10,
            "AeEnable": False,
            "ExposureTime": 80000,
            "AnalogueGain": 16.0,
            "AwbEnable": True,
            "AwbMode": controls.AwbModeEnum.Tungsten,
            "ColourGains": (1.8, 1.5),
            "Saturation": 1.2,
            "Sharpness": 2.5,
            "Contrast": 1.5,
            "Brightness": 0.2,
            "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality
        }
    )
