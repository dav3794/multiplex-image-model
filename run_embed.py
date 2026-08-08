import os
import numpy as np
import torch
from ruamel.yaml import YAML
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from glob import glob


from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.modules.immuvis import MultiplexAutoencoder
from multiplex_model.utils.configuration import EncoderConfig, DecoderConfig


models_path = "/raid_encrypted/immucan/models"
embeddings_path = "/raid_encrypted/immucan/embeddings"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
panel_config = "/home/mzmyslowski/marcin_multiplex/configs/all_panels_config.yaml"
tokenizer_config = "/home/mzmyslowski/marcin_multiplex/configs/all_markers_tokenizer.yaml"

PATCH_SIZE = 128
BATCH_SIZE = 1
NUM_WORKERS = 8
SAVE_EVERY = 200  # Save intermediate results every N images

print(f"Using device: {DEVICE}")


# Load configuration
yaml = YAML(typ="safe")
with open(panel_config, "r") as f:
    panel_config_dict = yaml.load(f)

panel_config_dict['datasets'] = ['hn']
# Override paths to raw TIFFs (all_panels_config.yaml points to pre-patched .npy used for training)
panel_config_dict['paths']['train'] = '/raid_encrypted/immucan/immuvis_split/train'
panel_config_dict['paths']['test'] = '/raid_encrypted/immucan/immuvis_split/test'

# Load tokenizer
with open(tokenizer_config, "r") as f:
    TOKENIZER = yaml.load(f)

# Create inverse tokenizer for channel names
INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
num_channels = len(TOKENIZER)

print(f"Number of channels: {num_channels}")
print(f"Sample markers: {list(TOKENIZER.keys())[:5]}")

# Create train and test datasets
train_dataset = DatasetFromTIFF(
    panels_config=panel_config_dict,
    split='train',
    marker_tokenizer=TOKENIZER,
    transform=None,
    use_median_denoising=False,
    use_butterworth_filter=True,
    use_minmax_normalization=False,
    use_global_clip_limits=False,
    use_clip_normalization=True,
)

test_dataset = DatasetFromTIFF(
    panels_config=panel_config_dict,
    split='test',
    marker_tokenizer=TOKENIZER,
    transform=None,
    use_median_denoising=False,
    use_butterworth_filter=True,
    use_minmax_normalization=False,
    use_global_clip_limits=False,
    use_clip_normalization=True,
)

train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE, shuffle=False)
test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=NUM_WORKERS)

print(f"Train dataset size: {len(train_dataset)} images")
print(f"Test dataset size: {len(test_dataset)} images")


def get_all_patches(img, patch_size: int = 128):
    """Extract all non-overlapping patches from an image."""
    H, W = img.shape[2:]
    i0, j0 = 0, 0
    i1, j1 = patch_size, patch_size
    patches = []
    coords = []
    
    while True:
        while True:
            patch = img[:, :, i0:i1, j0:j1]
            patches.append(patch)
            coords.append([(i0, j0), (i1, j1)])
            
            j1 += patch_size
            if j1 > W:
                break
            j0 = j1 - patch_size
        
        i1 += patch_size
        if i1 > H:
            break
        i0 = i1 - patch_size
        j0 = 0
        j1 = patch_size

    return patches, coords


def embed_images(
    model,
    dataloader, 
    device, 
    patch_size=128,
    outpath=None,
    split_name=None,
    model_prefix=None,
    save_every=200
):
    """Embed all images in the dataloader by extracting patches and encoding them."""
    model.eval()

    embeddings = []
    metadata = []
    batch_idx = 0
    
    for i, (img, channel_ids, panel_idx, img_path) in enumerate(tqdm(dataloader, desc=f"Embedding {split_name} images")):
        B, C, H, W = img.shape
        if H < patch_size or W < patch_size:
            print(f'Image is smaller than patch size: {img.shape} at {img_path[0]}')
            continue
        
        channel_ids = channel_ids.to(device)

        for patch, (coords0, coords1) in zip(*get_all_patches(img, patch_size)):
            patch = patch.to(torch.float32).to(device)
            metadata.append((os.path.realpath(img_path[0]), panel_idx[0], coords0, coords1))

            with torch.no_grad():
                latent = model.encode(patch, channel_ids)['output']
                embeddings.append(latent.cpu().numpy().squeeze(0))

        if (i + 1) % save_every == 0:
            print(f'Processed {i + 1} images, saving batch {batch_idx}...')
            # Save intermediate results
            if outpath:
                embeddings_array = np.stack(embeddings)
                np.save(
                    os.path.join(outpath, f'{model_prefix}_{split_name}_image_patches_embeddings_batch_{batch_idx}.npy'), 
                    embeddings_array
                )
                pd.DataFrame(
                    metadata, 
                    columns=['img_path', 'panel', 'coords0', 'coords1']
                ).to_csv(
                    os.path.join(outpath, f'{model_prefix}_{split_name}_image_patches_metadata_batch_{batch_idx}.csv'), 
                    index=False
                )

                embeddings = []
                metadata = []
                batch_idx += 1

    # Save remaining embeddings
    if embeddings:
        embeddings_array = np.stack(embeddings)
        np.save(
            os.path.join(outpath, f'{model_prefix}_{split_name}_image_patches_embeddings_batch_{batch_idx}.npy'), 
            embeddings_array
        )
        pd.DataFrame(
            metadata, 
            columns=['img_path', 'panel', 'coords0', 'coords1']
        ).to_csv(
            os.path.join(outpath, f'{model_prefix}_{split_name}_image_patches_metadata_batch_{batch_idx}.csv'), 
            index=False
        )
        print(f'Saved final batch {batch_idx}')
    
    print(f'Finished embedding {split_name} images!')

MODEL_WEIGHTS_PATH = "/home/mzmyslowski/marcin_multiplex/checkpoints/last_checkpoint-ImVs-25.pth"
MODEL_CONFIG_PATH = "/home/mzmyslowski/marcin_multiplex/train_masked_gp_marker_config_resume3.yaml"

for model_name in [MODEL_WEIGHTS_PATH]:
    model_checkpoint = os.path.basename(model_name)
    model_prefix = model_checkpoint.replace('.pth', '')

    print(f"\n{'='*80}")
    print(f"Processing model: {model_checkpoint}")
    print(f"{'='*80}")

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config_dict = yaml.load(f)

    encoder_config = EncoderConfig(**model_config_dict["encoder"])
    decoder_config = DecoderConfig(**model_config_dict["decoder"])

    # Initialize model
    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=encoder_config.model_dump(),
        decoder_config=decoder_config.model_dump(),
    ).to(DEVICE)

    # Load model weights
    print(f"Loading model weights from: {MODEL_WEIGHTS_PATH}")
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully!")

    
    # Embed test images
    print("\nEmbedding test dataset...")
    embed_images(
        model,
        test_dataloader, 
        DEVICE, 
        patch_size=PATCH_SIZE,
        outpath=embeddings_path,
        split_name='test',
        model_prefix=model_prefix,
        save_every=SAVE_EVERY
    )
    
    # Embed train images
    print("\nEmbedding train dataset...")
    embed_images(
        model,
        train_dataloader, 
        DEVICE, 
        patch_size=PATCH_SIZE,
        outpath=embeddings_path,
        split_name='train',
        model_prefix=model_prefix,
        save_every=SAVE_EVERY
    )
    
    print(f"\nCompleted embedding for {model_checkpoint}")
    
    # Clean up to free memory
    del model
    del checkpoint
    torch.cuda.empty_cache()

print(f"\n{'='*80}")
print("All models processed successfully!")
print(f"{'='*80}")
