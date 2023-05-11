import ffmpeg
import os

def time2str(time):
    pass 
    

def easy_seg(path, start, end, out_path):
    # ffmpeg -i input.mp4 -ss 00:00:03 -to 00:00:08 -c copy output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -t 5 -c copy output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -to 00:00:08 -async 1 -strict -2 output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -t 5 -async 1 -strict -2 output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -to 00:00:08 -c:v copy -c:a copy output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -t 5 -c:v copy -c:a copy output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -to 00:00:08 -c:v copy -c:a copy -async 1 -strict -2 output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -t 5 -c:v copy -c:a copy -async 1 -strict -2 output.mp4
    # ffmpeg -i input.mp4 -ss 00:00:03 -to 00:00:08 -c:v copy -c:a copy -bsf:v h264_mp4toannexb -f mpegts output.ts
    # ffmpeg -i input.mp4 -ss 00:00:03 -t 5 -c:v copy -c:a copy -bsf:v h264_mp4toannexb -f mpegts output.ts
    # ffmpeg -i input.mp4 -ss 00:00:03 -to 00:00:08 -c:v copy -c:a copy -bsf:v h264_mp4toannexb -f mpegts output.ts
    # ffmpeg -i input.mp4 -ss 00:00:03 -t 5 -c:v copy -c:a copy -bsf:v h264_mp4toannexb -f mpegts output.ts
    # ffmpeg -i input.mp4 -ss 00:
    os.popen('ffmpeg -i ' + path + ' -ss ' + start + ' -to ' + end + ' -c copy ' + out_path)
    