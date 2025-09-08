# FFmpeg with Vulkan Acceleration

## Building

FFmpeg Vulkan components require:
* Vulkan development environment (FFmpeg decoders and encoders)
* GLSL/HLSL to SPIRV shaders compiler (FFmpeg filters)

On Ubuntu the minimal build environment can be installed by:

```
sudo apt-get install \
  gcc libshaderc-dev libtool libvulkan-dev make nasm pkg-config
```

Then FFmpeg can be configured and built as follows (explicit `--enable-libshaderc` is required):

```
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg && cd ffmpeg
./configure \
     --prefix=$PREFIX \
     --libdir=$LIBDIR \
     --disable-static \
     --disable-stripping \
     --disable-doc \
     --enable-shared \
     --enable-vulkan \
     --enable-libshaderc
make -j $(nproc)
make install
```

## Running

Vulkan Mesa driver for Intel GPUs (ANV) might require the following environment variables to enable Vulkan video capabilities:

```
export ANV_VIDEO_DECODE=1
export ANV_VIDEO_ENCODE=1
```

Vulkan driver stack is sensitive to `DISPLAY` environment variable. Unsetting `DISPLAY` will reduce driver capabilities to
those not requiring on screen rendering. Media decoding, encoding and conversion should still work.

To test decoding with FFmpeg, generate or download input stream. For example, to test H264 decoder
https://github.com/pytorch/torchcodec/blob/v0.7.0/test/resources/nasa_13013.mp4 can be used.

### Basic checks

Use `vulkaninfo` to check that driver actually exposes video capabilities:

```
vulkaninfo | grep VK_KHR_video
'DISPLAY' environment variable not set... skipping surface info
        VK_KHR_video_decode_av1                       : extension revision 1
        VK_KHR_video_decode_h264                      : extension revision 9
        VK_KHR_video_decode_h265                      : extension revision 8
        VK_KHR_video_decode_queue                     : extension revision 8
        VK_KHR_video_encode_h264                      : extension revision 14
        VK_KHR_video_encode_h265                      : extension revision 14
        VK_KHR_video_encode_queue                     : extension revision 12
        VK_KHR_video_maintenance1                     : extension revision 1
        VK_KHR_video_queue                            : extension revision 8
```

Use the following FFmpeg command line to initialize Vulkan within FFmpeg and exit. Debug logs will list all available Vulkan
devuces, the one which got selected and its basic information.

```
$ ffmpeg -v debug -init_hw_device vulkan
ffmpeg version N-120972-g6696a9b8bd Copyright (c) 2000-2025 the FFmpeg developers
  built with gcc 14 (Ubuntu 14.2.0-19ubuntu2)
  configuration: --prefix=/localdisk/dmitry/_install --libdir=/localdisk/dmitry/_install/lib --disable-static --disable-stripping --disable-doc --enable-shared --enable-vaapi --enable-libmp3lame --enable-gpl --enable-libx264 --enable-libvpl --enable-libshaderc --enable-vulkan
  libavutil      60. 12.100 / 60. 12.100
  libavcodec     62. 15.100 / 62. 15.100
  libavformat    62.  4.102 / 62.  4.102
  libavdevice    62.  2.100 / 62.  2.100
  libavfilter    11.  8.100 / 11.  8.100
  libswscale      9.  3.100 /  9.  3.100
  libswresample   6.  2.100 /  6.  2.100
Splitting the commandline.
Reading option '-v' ... matched as option 'v' (set logging level) with argument 'debug'.
Reading option '-init_hw_device' ... matched as option 'init_hw_device' (initialise hardware device) with argument 'vulkan'.
Finished splitting the commandline.
Parsing a group of options: global .
Applying option v (set logging level) with argument debug.
Applying option init_hw_device (initialise hardware device) with argument vulkan.
[Vulkan @ 0x61c7ac35bd40] Supported layers:
[Vulkan @ 0x61c7ac35bd40]       VK_LAYER_MESA_device_select
[Vulkan @ 0x61c7ac35bd40]       VK_LAYER_MESA_overlay
[Vulkan @ 0x61c7ac35bd40]       VK_LAYER_INTEL_nullhw
[Vulkan @ 0x61c7ac35bd40] GPU listing:
[Vulkan @ 0x61c7ac35bd40]     0: Intel(R) Graphics (BMG G21) (discrete) (0xe20b)
[Vulkan @ 0x61c7ac35bd40]     1: Intel(R) Graphics (RPL-S) (integrated) (0xa780)
[Vulkan @ 0x61c7ac35bd40]     2: llvmpipe (LLVM 19.1.7, 256 bits) (software) (0x0)
[Vulkan @ 0x61c7ac35bd40] Device 0 selected: Intel(R) Graphics (BMG G21) (discrete) (0xe20b)
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_push_descriptor
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_EXT_physical_device_drm
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_EXT_shader_atomic_float
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_cooperative_matrix
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_shader_subgroup_rotate
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_shader_expect_assume
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_maintenance1
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_external_memory_fd
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_EXT_external_memory_dma_buf
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_EXT_image_drm_format_modifier
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_external_semaphore_fd
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_EXT_external_memory_host
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_queue
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_encode_queue
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_decode_queue
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_encode_h264
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_decode_h264
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_encode_h265
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_decode_h265
[Vulkan @ 0x61c7ac35bd40] Using device extension VK_KHR_video_decode_av1
[Vulkan @ 0x61c7ac35bd40] Queue families:
[Vulkan @ 0x61c7ac35bd40]     0: graphics compute transfer sparse (queues: 1)
[Vulkan @ 0x61c7ac35bd40]     1: compute transfer sparse (queues: 1)
[Vulkan @ 0x61c7ac35bd40]     2: encode decode (queues: 2)
[Vulkan @ 0x61c7ac35bd40]     3: transfer (queues: 1)
[Vulkan @ 0x61c7ac35bd40] Using device: Intel(R) Graphics (BMG G21)
[Vulkan @ 0x61c7ac35bd40] Alignments:
[Vulkan @ 0x61c7ac35bd40]     optimalBufferCopyRowPitchAlignment: 128
[Vulkan @ 0x61c7ac35bd40]     minMemoryMapAlignment:              4096
[Vulkan @ 0x61c7ac35bd40]     nonCoherentAtomSize:                64
[Vulkan @ 0x61c7ac35bd40]     minImportedHostPointerAlignment:    4096
[Vulkan @ 0x61c7ac35bd40] Using queue family 0 (queues: 1) for graphics
[Vulkan @ 0x61c7ac35bd40] Using queue family 1 (queues: 1) for compute
[Vulkan @ 0x61c7ac35bd40] Using queue family 3 (queues: 1) for transfers
[Vulkan @ 0x61c7ac35bd40] Using queue family 2 (queues: 2) for encode decode
Successfully parsed a group of options.
Universal media converter
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...

Use -h to get full help or, even better, run 'man ffmpeg'

Exiting with exit code 1
```

To initialize specific Vulkan device use `-init_hw_device vulkan=gpu:N` option with `N=0|1|2|...`

### Decoding

Decode stream and discard the output:

```
ffmpeg -hwaccel vulkan -init_hw_device vulkan=gpu:0 -hwaccel_output_format vulkan \
  -i nasa_13013.mp4 -f null -
```

Decode stream and save the output:

```
ffmpeg -hwaccel vulkan -init_hw_device vulkan=gpu:0 -hwaccel_output_format vulkan \
  -i nasa_13013.mp4 -vf "hwdownload,format=nv12" -y nasa_13013.yuv
```

Decode stream and scale the output:

```
ffmpeg -hwaccel vulkan -init_hw_device vulkan=gpu:0 -hwaccel_output_format vulkan \
  -i nasa_13013.mp4 -vf "scale_vulkan=1280:720,hwdownload,format=nv12" -y nasa_13013_1280x720.yuv
```

### Transcoding

> [!CAUTION]
> Encoding (or transcoding) did not work for me on Intel driver on Battle Image GPU causing
> GPU hangs.

Transcode H264 to H265:

```
ffmpeg -hwaccel vulkan -init_hw_device vulkan=gpu:0 -hwaccel_output_format vulkan \
  -i nasa_13013.mp4 -c:v h264_vulkan nasa_13013.h265.mp4
```
