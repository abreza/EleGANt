import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args

def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    source = Image.open(args.source).convert('RGB')
    lip_ref = Image.open(args.lip_ref).convert('RGB')
    skin_ref = Image.open(args.skin_ref).convert('RGB')
    eye_ref = Image.open(args.eye_ref).convert('RGB')

    result = inference.joint_transfer(source, lip_ref, skin_ref, eye_ref, postprocess=True) 
    if result is None:
        return
    
    source = np.array(source)
    lip_ref = np.array(lip_ref)
    skin_ref = np.array(skin_ref)
    eye_ref = np.array(eye_ref)

    h, w, _ = source.shape
    result = result.resize((h, w)); result = np.array(result)
    vis_image = np.hstack((source, lip_ref, skin_ref, eye_ref, result))
    save_path = os.path.join(args.save_folder, "result.png")
    Image.fromarray(vis_image.astype(np.uint8)).save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default='ckpts/sow_pyramid_a5_e3d2_remapped.pth')

    parser.add_argument("--source", type=str, default="assets/images/non-makeup/source_1.png")
    parser.add_argument("--lip_ref", type=str, default="assets/images/makeup/reference_1.png")
    parser.add_argument("--skin_ref", type=str, default="assets/images/makeup/reference_2.png")
    parser.add_argument("--eye_ref", type=str, default="assets/images/makeup/reference_3.png")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)