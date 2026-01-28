#!/usr/bin/env python3
# generate_clinical.py
# -*- coding: utf-8 -*-


import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from modules_vit import MultiplexAutoencoder
from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler


sys.path.insert(0, os.getcwd())
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

def get_all_patches(img, patch_size=128):
    if img.ndim == 3: img = img.unsqueeze(0)
    
    H, W = img.shape[2:]
    patches = []
    coords = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            if i + patch_size > H or j + patch_size > W:
                continue

            patch = img[:, :, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
            coords.append([(i, j), (i + patch_size, j + patch_size)])

    return patches, coords


def load_model_from_config(config_path, checkpoint_path, device):
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    encoder_conf = config['encoder']
    decoder_conf = config['decoder']
    num_channels = 265
    print(f"[DEBUG]: num_channels = {num_channels} (from checkpoints)")
    input_size = config.get('input_image_size', [128, 128])
    model_config = {
        'num_channels': num_channels,
        'encoder_config': encoder_conf,
        'decoder_config': decoder_conf,
        'encoder_type': encoder_conf.get('encoder_type', 'convnext'),
        'input_image_size': tuple(input_size)
    }

    for key in ['vit_config', 'swin_config', 'dinov2_config', 'dinov3_config']:
        if key in encoder_conf:
            model_config[key] = encoder_conf[key]

    model = MultiplexAutoencoder(**model_config)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Success: Weights loaded with strict=True")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed. Retrying with strict=False.")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def process_dataset(model, dataloader, device, patch_size, outpath, split_name, model_name):
    BATCH_SAVE_FREQ = 500
    TARGET_SUBSETS = ['danenberg', 'cords']

    embeddings = []
    metadata = []    
    batch_idx = 0
    images_processed_count = 0

    for img, channel_ids, panel_idx, img_path_tuple in tqdm(dataloader, desc=split_name):
        img_path = img_path_tuple[0]
        
        if not any(subset in img_path for subset in TARGET_SUBSETS):
            continue

        _, _, H, W = img.shape
        if H < patch_size or W < patch_size:
            continue

        channel_ids = channel_ids.to(device)
        patches, coords = get_all_patches(img, patch_size)
        
        if len(patches) == 0:
            continue

        inference_batch_size = 64
        for i in range(0, len(patches), inference_batch_size):
            batch_patches = patches[i:i+inference_batch_size]
            batch_coords = coords[i:i+inference_batch_size]
            x = torch.cat(batch_patches, dim=0).float().to(device)
            curr_B = x.shape[0]
            batch_c_ids = channel_ids.repeat(curr_B, 1)

            with torch.no_grad():
                enc_out = model.encode(x, batch_c_ids)
                latent = enc_out['output']

                if latent.ndim == 4:
                    latent = latent.mean(dim=(2, 3)) 
                
                latent_np = latent.cpu().numpy()
                
            for k in range(curr_B):
                embeddings.append(latent_np[k])
                c0, c1 = batch_coords[k]
                c0_str = f"({c0[0]}, {c0[1]})"
                c1_str = f"({c1[0]}, {c1[1]})"
                p_idx = panel_idx.item() if isinstance(panel_idx, torch.Tensor) else panel_idx
                metadata.append((img_path, p_idx, c0_str, c1_str))

        images_processed_count += 1

        if images_processed_count % BATCH_SAVE_FREQ == 0:
            save_batch(embeddings, metadata, outpath, split_name, model_name, batch_idx)
            batch_idx += 1
            embeddings = []
            metadata = []

    if len(embeddings) > 0:
        save_batch(embeddings, metadata, outpath, split_name, model_name, batch_idx)


def save_batch(embeddings, metadata, outpath, split_name, model_name, batch_idx):
    if not embeddings:
        return None
    
    emb_arr = np.stack(embeddings)
    save_base = f"ImmuVis-{model_name}_{split_name}_patches"
    npy_path = os.path.join(outpath, f'{save_base}_embeddings_batch_{batch_idx}.npy')
    csv_path = os.path.join(outpath, f'{save_base}_metadata_batch_{batch_idx}.csv')
    np.save(npy_path, emb_arr)
    df = pd.DataFrame(metadata, columns=['img_path', 'panel', 'coords0', 'coords1'])
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, help="Short name for files e.g. ViTS")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--panel_config', type=str, default='/YOUR/PATH/all_panels_config2.yaml')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load model
    model = load_model_from_config(args.config, args.checkpoint, device)

    # Dataset
    tokenizer_path = '/YOUR/PATH/all_markers_tokenizer.yaml'
    if os.path.exists(tokenizer_path):
        tokenizer = YAML().load(open(tokenizer_path))
    else:
        if os.path.exists('configs/all_markers_tokenizer.yaml'):
            tokenizer = YAML().load(open('configs/all_markers_tokenizer.yaml'))
        else:
            print(f"[WARNING]: Tokenizer not found at {tokenizer_path} nor locally. Ensure path is correct.")
            sys.exit(1)
    
    if os.path.exists(args.panel_config):
        panel_config = YAML().load(open(args.panel_config))
    else:
        print(f"Error: Panel config not found at {args.panel_config}")
        sys.exit(1)

    # Dataset Test
    test_dataset = DatasetFromTIFF(
        panels_config=panel_config,
        split='test',
        marker_tokenizer=tokenizer,
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_global_clip_limits=False,
        use_clip_normalization=True,
    )
    test_loader = DataLoader(test_dataset, batch_sampler=PanelBatchSampler(test_dataset, 1, shuffle=False), num_workers=8)

    # Dataset Train
    train_dataset = DatasetFromTIFF(
        panels_config=panel_config,
        split='train',
        marker_tokenizer=tokenizer,
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_global_clip_limits=False,
        use_clip_normalization=True,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=PanelBatchSampler(train_dataset, 1, shuffle=False), num_workers=8)

    process_dataset(model, test_loader, device, 128, model_output_dir, 'test', args.model_name)
    process_dataset(model, train_loader, device, 128, model_output_dir, 'train', args.model_name)

if __name__ == '__main__':
    main()
