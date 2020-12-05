# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from cyclegan_pytorch import Generator
import pdb 


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--file", type=str, help="Video name.")
    parser.add_argument("--model-name", type=str, default="weights/horse2zebra/netG_A2B.pth",
                        help="dataset name.  (default:`weights/horse2zebra/netG_A2B.pth`).")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--image-size", type=int, default=256,
                        help="size of the data crop (squared assumed). (default:256)")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if args.cuda else "cpu")

    # create model
    model = Generator().to(device)

    # Load state dicts
    model.load_state_dict(torch.load(args.model_name))

    # Set model mode
    model.eval()

    # Load video
    videoCapture = cv2.VideoCapture(args.file)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (w, h)

    out_video_writer = cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')  
    video_name = os.path.basename(args.file).split('.')[0]
    output_video_name = "demo/out_" + video_name + ".avi"
    out_video_writer.open(output_video_name, fourcc, fps, video_size, isColor=True)

    pre_process = transforms.Compose([transforms.Resize(args.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # read frame
    success, frame = videoCapture.read()
    test_bar = tqdm(range(int(frame_numbers)), 
                        desc="[processing video and saving result videos]")
    frame_count = 0
    for index in test_bar:
        if success:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = pre_process(image).unsqueeze(0)

            image = image.to(device)

            out = model(image)
            out = out.cpu() # (1, 3, args.image_size, args.image_size)
            out_image = out.data[0].numpy()  # (3, args.image_size, args.image_size)
            out_image = deprocess_image(out_image)
            out_image = out_image.transpose((1, 2, 0))  # (chw)-> (hwc)
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
            out_image = cv2.resize(out_image, (w,h))
            # save out_video
            frame_count += 1
            # cv2.imwrite('demo/frame{:05d}.jpg'.format(frame_count), out_image)
            out_video_writer.write(out_image)

            success, frame = videoCapture.read()

    out_video_writer.release()
    videoCapture.release()

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    main()