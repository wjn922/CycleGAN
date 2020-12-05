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
import random
import time

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

from cyclegan_pytorch import Generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--file", type=str, default="demo/horse.png",
                        help="Image name. (default:`demo/horse.png`)")
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

    # Load image
    image = Image.open(args.file).convert('RGB')
    pre_process = transforms.Compose([transforms.Resize(args.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                      ])
    image = pre_process(image).unsqueeze(0)  # (1, 3, H, W)
    image = image.to(device)

    start = time.time()
    fake_image = model(image)
    elapsed = (time.time() - start)
    print(f"cost {elapsed:.4f}s")
    vutils.save_image(fake_image.detach(), "demo/result.png", normalize=True)

if __name__ == '__main__':
    main()
