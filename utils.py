import os
import subprocess

import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import tensorflow.compat.v1 as tf
import cv2
from hiFill.utils import *

from spleeter.separator import Separator

def mp42img(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    if os.path.isfile(output_path + '/' + 'frame_1.png'):
        print('mp42img is already done!')
    else:
        command = 'ffmpeg -i ' + input_path + ' -vcodec png -r 30 ' + output_path + '/' + 'frame_%d.png'
        subprocess.call(command, shell=True)
        print("Success mp4 to img")


def segmentation(input_path, output_path):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()

    os.makedirs(output_path, exist_ok=True)
    if os.path.isfile(output_path + '/' + 'frame_1.png'):
        print('semantic segmentation is already done!')
    else:

        input_num = sum(os.path.isfile(os.path.join(input_path, name)) for name in os.listdir(input_path))

        for file_name in tqdm(os.listdir(input_path), total=input_num):
            input_image = Image.open(input_path + '/' + file_name).convert('RGB')
            preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)

            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            else:
                input_batch = input_batch.to('cpu')
                model.to('cpu')

            with torch.no_grad():
                outputs = model(input_batch)['out'][0]
                output_predictions = outputs.argmax(0)

            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")
  
            for i in range(21):
                colors[i][0] = 255
                colors[i][1] = 255
                colors[i][2] = 255

            #person = 15
            colors[15][0] = 0
            colors[15][1] = 0
            colors[15][2] = 0

            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)

            if r.mode != "RGB":
                r = r.convert("RGB")
            r.save(output_path + "/" + file_name)


def inpainting(frame_path, mask_path, output_path):
    paths_img, paths_mask, len_imgs = read_imgs_masks(frame_path, mask_path)

    os.makedirs(output_path, exist_ok=True)
    if os.path.isfile(output_path + '/' + 'frame_1.png'):
        print('inpainting is already done!')
    else:
        with tf.Graph().as_default():
            with open('hiFill/pb/hifill.pb', "rb") as f:
                output_graph_def = tf.GraphDef()
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")

            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                image_ph = sess.graph.get_tensor_by_name('img:0')
                mask_ph = sess.graph.get_tensor_by_name('mask:0')
                inpainted_512_node = sess.graph.get_tensor_by_name('inpainted:0')
                attention_node = sess.graph.get_tensor_by_name('attention:0')
                mask_512_node = sess.graph.get_tensor_by_name('mask_processed:0')

                for path_img, path_mask in tqdm(zip(paths_img, paths_mask), total=len_imgs):
                    raw_img   = cv2.imread(path_img)
                    raw_mask  = cv2.imread(path_mask)
                    inpainted = inpaint(raw_img, raw_mask, sess, inpainted_512_node, attention_node, mask_512_node, image_ph, mask_ph, 6)
                    filename  = output_path + '/' + os.path.basename(path_img).split('.')[0]
                    cv2.imwrite(filename + '.png', inpainted)


def img2mp4(input_path, output_path):
    file_num = sum(os.path.isfile(os.path.join(input_path, name)) for name in os.listdir(input_path))
    command = 'ffmpeg -r 30 -start_number 0 -i ' + input_path + '/' + 'frame_%d.png -vframes ' + str(file_num) + ' -vcodec libx264 -pix_fmt yuv420p -r 30 ' + output_path + '/results.mp4'
    subprocess.call(command, shell=True)
    print("Success img to mp4")


def spleeter(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    if os.path.isfile(output_path + '/' + 'audio.mp3'):
        print('audio-separate is already done!')
    else:
        command = 'ffmpeg -i ' + input_path + ' -f mp3 -ar 44100 -vn ' + output_path + '/audio.mp3'
        subprocess.call(command, shell=True)

    input_audio = output_path + '/audio.mp3'
    separator_2stem = Separator('spleeter:2stems')
    separator_2stem.separate_to_file(input_audio, output_path)