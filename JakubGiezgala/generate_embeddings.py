#!/usr/bin/env python3
# generate_embeddings.py
# -*- coding: utf-8 -*-


import os
import gc
import sys
import glob
import torch
import argparse
import numpy as np

from tqdm import tqdm
from ruamel.yaml import YAML
from torch.nn.functional import interpolate
from modules_vit import MultiplexAutoencoder

sys.path.insert(0, os.getcwd()) 


def load_model(config_path, checkpoint_path, device):
    print(f"Loading model from {checkpoint_path} with config {config_path}...")
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    # Encoder from CONFIG
    encoder_conf = config['encoder'].copy()

    encoder_type = encoder_conf.get('encoder_type', 'convnext')
    vit_config = encoder_conf.get('vit_config', None)
    swin_config = encoder_conf.get('swin_config', None)
    dinov2_config = encoder_conf.get('dinov2_config', None)
    dinov3_config = encoder_conf.get('dinov3_config', None)
    
    decoder_conf = config['decoder']
    tokenizer_path = config.get('tokenizer_config', 'configs/all_markers_tokenizer.yaml')
    
    # Channels
    if os.path.exists(tokenizer_path):
        tokenizer = YAML().load(open(tokenizer_path))
        num_channels = len(tokenizer)
    elif os.path.exists('configs/all_markers_tokenizer.yaml'):
        tokenizer = YAML().load(open('configs/all_markers_tokenizer.yaml'))
        num_channels = len(tokenizer)
    else:
        num_channels = 265 

    input_size = config.get('input_image_size', [128, 128])

    model_config = {
        'num_channels': num_channels,
        'encoder_config': encoder_conf,
        'decoder_config': decoder_conf,
        'encoder_type': encoder_type,
        'vit_config': vit_config,
        'swin_config': swin_config,
        'dinov2_config': dinov2_config,
        'dinov3_config': dinov3_config,
        'input_image_size': tuple(input_size)
    }

    model = MultiplexAutoencoder(**model_config).to(device)
    
    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    else:
        state_dict = checkpoint
        
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded strictly.")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed ({str(e)[:100]}...). Retrying with strict=False.")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, tuple(input_size)


def process_batch(model, batch_imgs, batch_ids, device, target_size):
    x = torch.from_numpy(batch_imgs).float().to(device)
    # Interpolation (e.g. 32 -> 128)
    if x.shape[-2:] != target_size:
        x = interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    B, C, _, _ = x.shape

    if batch_ids is not None:
        encoded_indices = torch.from_numpy(batch_ids).long().to(device)
        if encoded_indices.ndim == 1:
             encoded_indices = encoded_indices.unsqueeze(0).repeat(B, 1)
    else:
        encoded_indices = torch.arange(C, device=device).unsqueeze(0).repeat(B, 1)

    with torch.no_grad():
        # Inference
        enc_out = model.encode(x, encoded_indices)
        latent = enc_out['output']
        
        # Global Average Pooling [B, C, H, W] -> [B, C]
        if latent.ndim == 4:
            embeddings = latent.mean(dim=(2, 3))
        elif latent.ndim == 3: 
             embeddings = latent.mean(dim=1) 
        else:
             embeddings = latent
        
    return embeddings.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings (Singular Mode)")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--input_dir', type=str, nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    files_to_process = []
    for d in args.input_dir:
        files_to_process.extend(glob.glob(os.path.join(d, "*.npy")))
        files_to_process.extend(glob.glob(os.path.join(d, "*.npz")))
        
    # Filter output files
    files_to_process = [f for f in files_to_process if not os.path.basename(f).startswith('emb_')]
    
    print(f"Found {len(files_to_process)} files to process.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    try:
        model, target_size = load_model(args.config, args.checkpoint, device)
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        sys.exit(1)

    # Process Files
    for file_path in tqdm(files_to_process, desc="Files"):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        save_name = f"emb_{base_name}.npy"
        save_path = os.path.join(args.output_dir, save_name)
        
        if os.path.exists(save_path):
            continue

        try:
            # Load data
            data_raw = np.load(file_path, allow_pickle=True)
            
            if isinstance(data_raw, np.ndarray) and data_raw.ndim == 0:
                data = data_raw.item()
            elif isinstance(data_raw, np.lib.npyio.NpzFile):
                data = data_raw
            else:
                data = data_raw 

            if 'patches' not in data:
                continue

            patches = data['patches']
            labels = data['labels'] if 'labels' in data else None
            channel_ids = data['channel_ids'] if 'channel_ids' in data else None
            
            # Fix Dimensions [N, H, W, C] -> [N, C, H, W]
            if patches.ndim == 4:
                 if patches.shape[-1] < patches.shape[-2] and patches.shape[-1] < patches.shape[-3]:
                      patches = patches.transpose(0, 3, 1, 2)
            
            num_samples = patches.shape[0]
            if num_samples == 0:
                continue

            file_embeddings = []
            
            # Batch Loop
            for i in range(0, num_samples, args.batch_size):
                batch_imgs = patches[i : i + args.batch_size]
                emb = process_batch(model, batch_imgs, channel_ids, device, target_size)
                file_embeddings.append(emb)

            # Save
            if len(file_embeddings) > 0:
                final_embeddings = np.concatenate(file_embeddings, axis=0)
                output_data = {'embeddings': final_embeddings}
                if labels is not None:
                    output_data['labels'] = labels
                
                np.save(save_path, output_data)

            del patches, file_embeddings, batch_imgs, data
            if 'output_data' in locals(): del output_data
                
        except Exception as e:
            print(f"Error in {file_name}: {e}")

        gc.collect()

if __name__ == '__main__':
    main()
