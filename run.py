import os
import argparse
import shutil
import subprocess
from utils import *

cline = argparse.ArgumentParser(description='directory reference')
cline.add_argument('-input_file', default='./input/test.mp4', help='path to input video')
cline.add_argument('-output_dir', default='./output', help='path to output results')

if __name__ == "__main__":
    args = cline.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    input_video = args.input_file
    video_name, ext = os.path.splitext(os.path.basename(input_video))
    output_path = args.output_dir + '/' +video_name
    os.makedirs(output_path, exist_ok=True)

    frame_path   = output_path + '/' + 'frames'
    mask_path    = output_path + '/' + 'mask'
    inpaint_path = output_path + '/' + 'inpaint'

    mp42img(input_video, frame_path)
    segmentation(frame_path, mask_path)
    inpainting(frame_path, mask_path, inpaint_path)

    img2mp4(inpaint_path, output_path)

    audio_path    = output_path + '/' + 'audio'
    spleeter_path = output_path + '/' + 'audio' + '/' + 'audio'

    spleeter(input_video, audio_path)

    output_name = args.output_dir + '/' +video_name + '.mp4'

    command = 'ffmpeg -i ' + output_path + '/results.mp4 -i ' + spleeter_path + '/accompaniment.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 ' + output_name
    subprocess.call(command, shell=True)
    shutil.rmtree(output_path)
    print(output_name + " : created!!")

