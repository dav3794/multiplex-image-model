import os
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import neptune
from neptune.utils import stringify_unsupported
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import RandomRotation, RandomCrop
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple
from torchvision import transforms


from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import nll_loss
from multiplex_model.utils import ClampWithGrad, plot_reconstructs_with_uncertainty, get_scheduler_with_warmup
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.run_utils import build_run_name_suffix, SLURM_JOB_ID

import sys
import os


from src_from_rudy.utils import setup_seeds, setup_clearml, ChainDataLoader
from src_from_rudy.augmentations import ArcsinhNormalize, MinMaxNormalize, ButterworthFilter, GlobalNormalize, SelectMarkers
from src_from_rudy.config.autoencoder_cells import Config as CellsConfig
from src_from_rudy.config.multiplex_vit_no_bottleneck_autoencoder import Config as TissuesConfig
from src_from_rudy.constants import LOGDIR, IMC_PANEL2_DATA_DIR, PANEL_1_MARKER_NAMES, IMC_PANEL1_DATA_DIR, HALEY_GLIO, DANENBERG_BREAST_DIR, REQUIRED_IMG_SHAPES
from src_from_rudy.datasets.imc_datasets import CellPatchesDataModule, WholeSlidesDataModule, ContrastiveTissuePatchesDataModule, TissuePatchesDataModule



def _setup_data_loaders(data_module) -> Tuple[DataLoader, DataLoader]:
    data_module.setup('train')
    data_module.setup('test')
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    return train_loader, test_loader

def _setup_data_loader(data_module) -> DataLoader:
    data_module.setup('train')
    train_loader = data_module.train_dataloader()
    return train_loader

def _get_marker_indices(markers):
    marker_indices = []
    for marker in markers:
        marker_indices.append(PANEL_1_MARKER_NAMES.index(marker))
    return marker_indices

def _create_tissue_patches_data_module(data_dir: str, batch_size: int, image_size: int):
    # patches_transform = transforms.Compose([
    #     ArcsinhNormalize(),
    #     MinMaxNormalize(),
    #     ButterworthFilter(),

    # ])
    patches_transform = transforms.Compose([
        # ArcsinhNormalize(),
        # ButterworthFilter(),
        # GlobalNormalize(),
        # SelectMarkers(_get_marker_indices(TissuesConfig.SELECTED_MARKERS)),
    ])
    patches_data_module = TissuePatchesDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        patch_size=image_size,
        transform=patches_transform,
        sample_frac=TissuesConfig.SAMPLE_FRAC,
        img_paths_file=TissuesConfig.IMG_PATHS_FILE,
        is_imc_dir=data_dir in [IMC_PANEL2_DATA_DIR, IMC_PANEL1_DATA_DIR],
        need_to_merge_single_channel_images=data_dir in [HALEY_GLIO],
        required_img_shape=REQUIRED_IMG_SHAPES[data_dir],
    )
    return patches_data_module




def train_masked(
        model, 
        optimizer,
        scheduler,
        train_dataloader, 
        val_dataloader, 
        device, 
        run,
        marker_names_map,
        epochs=10, 
        gradient_accumulation_steps=1,
        min_channels_frac=0.5,
        start_epoch=0,
        save_checkpoint_every=50,
        checkpoints_path='checkpoints'
    ):
    """Train a masked autoencoder (decode the remaining channels) with the given parameters."""
    model.train()
    scaler = GradScaler()
    run_name = run['sys/name'].fetch()

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f'Created checkpoints directory at {checkpoints_path}')

    # val_loss = test_masked(
    #     model, 
    #     val_dataloader, 
    #     device, 
    #     run, 
    #     0, 
    #     min_channels_frac=min_channels_frac,
    #     marker_names_map=marker_names_map,
    # )
    # print(f'Validation loss: {val_loss:.4f}')
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):
            img = batch['image']
            # crop image to first 99x99 pixels
            # img = img[:, :, :97, :97]  # Assuming img is of shape [batch_size, num_channels, H, W]
            img = img[:, :, 1:-2, 1:-2]
            # print(f'Processing batch {batch_idx} in epoch {epoch}...')
            # print(f'Batch size: {img.shape[0]}, Image shape: {img.shape}, Channel IDs shape: {channel_ids.shape}, Panel index: {panel_idx}, Image path: {img_path}')
            batch_size, num_channels, H, W = img.shape
            # trim channel_ids for debugging to 10 first channels
            # channel_ids = channel_ids[:, :10]
            channel_ids = torch.arange(num_channels, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, num_channels]

            # Randomly sample a subset of channels to keep
            min_channels = int(np.ceil(num_channels * min_channels_frac))

            num_sampled_channels = np.random.randint(min_channels, num_channels)
            channels_subset_idx = [
                np.random.choice(
                    np.arange(num_channels), 
                    size=(1, num_sampled_channels), 
                    replace=False
                ) for _ in range(batch_size)
            ]

            channels_subset_indices = np.concatenate(channels_subset_idx, axis=0)
            channels_subset_indices = torch.tensor(channels_subset_indices, dtype=torch.long)

            channels_subset_indices = channels_subset_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # Shape: [batch_size, num_sampled_channels, H, W]
            masked_img = torch.gather(img, dim=1, index=channels_subset_indices).to(device) 
            # print(f'Batch {batch_idx} - Masked image shape: {masked_img.shape}, Channels subset indices shape: {channels_subset_indices.shape}')

            # Gather corresponding channel IDs
            active_channel_ids = torch.gather(channel_ids, dim=1, index=channels_subset_indices[:, :, 0, 0]).to(device) # Shape: [batch_size, num_sampled_channels]


            img = img.to(device)
            print(f"img shape {img.shape}")
            channel_ids = channel_ids.to(device)
            # print(f"Batch {batch_idx} - Channel IDs: {channel_ids.shape}  Active channel IDs: {active_channel_ids.shape}, Masked image shape: {masked_img.shape}, Image shape: {img.shape}")
            masked_img = masked_img.to(torch.float32)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # output = model(masked_img, active_channel_ids, channel_ids)['output']
                output = model(masked_img, active_channel_ids, channel_ids)['output'][:, :, 3:-4, 3:-4]
                # output = model(masked_img, active_channel_ids, active_channel_ids)['output'][:, :, 3:-4, 3:-4]
                # output = model(masked_img, active_channel_ids, active_channel_ids)[0][:, :, 3:-4, 3:-4]
                # print(f"output shape: {output.shape}")
                mi, logsigma = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)
                print(f'Mean of mi: {mi.mean().item()}, Mean of logsigma: {logsigma.mean().item()}')

                # Apply ClampWithGrad to logsigma for stability
                # logsigma = ClampWithGrad.apply(logsigma, -15.0, 15.0)
                logsigma = torch.tanh(logsigma) * 5.0  # Scale logsigma to a reasonable range
                loss = nll_loss(img, mi, logsigma)
                print(loss.item())

                # sanity check if loss is finite
                # panel_idx = "nsclc-panel-1"  # Placeholder for panel index, can be modified based on your dataset
                # img_path = train_dataloader.dataset.curr_img_path
                if not torch.isfinite(loss):
                    print(f'Non-finite loss encountered at batch {batch_idx} in epoch {epoch}. Skipping batch.')
                    print(f'Dataset: {panel_idx}')
                    print(f'Image path: {img_path}')
                    print(f'Mi: {mi}, Logsigma: {logsigma}')
                    assert False, "Non-finite loss encountered. Check the model and data."

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx+1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler.step()
                run['train/loss'].append(loss.item())
                run['train/lr'].append(scheduler.get_last_lr()[0])
                run['train/µ'].append(mi.mean().item())
                run['train/logvar'].append(logsigma.mean().item())
                run['train/mae'].append(torch.abs(img - mi).mean().item())
        scheduler.step()

        val_loss = test_masked(
            model, 
            val_dataloader, 
            device, 
            run, 
            epoch, 
            min_channels_frac=min_channels_frac,
            marker_names_map=marker_names_map,
        )
        print(f'Validation loss: {val_loss:.4f}')

        if (epoch + 1) % save_checkpoint_every == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, f'{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth')

    final_model_path = f'{checkpoints_path}/final_model-{run_name}.pth'
    print(f'Training completed. Saving final model at {final_model_path}...')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epochs,
    }
    torch.save(checkpoint, final_model_path)


def test_masked(
        model,  
        test_dataloader, 
        device, 
        run, 
        epoch,
        marker_names_map,
        num_plots=10, 
        min_channels_frac=0.5
        ):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    plot_indices = np.random.choice(np.arange(len(test_dataloader)), size=num_plots, replace=False)
    plot_indices = set(plot_indices)
    rand_gen = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader, desc=f'Testing epoch {epoch}')):
            img = batch['image']
            # img = img[:, :, :97, :97]  # Crop image to first 97x97 pixels
            img = img[:, :, 1:-2, 1:-2]
            batch_size, num_channels, H, W = img.shape
            min_channels = int(np.ceil(num_channels * min_channels_frac))

            channel_ids = torch.arange(num_channels, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, num_channels]
            
            num_sampled_channels = torch.randint(min_channels, num_channels, (1,), generator=rand_gen).item()
            channels_subset_idx = [
                np.random.choice(
                    np.arange(num_channels), 
                    size=(1, num_sampled_channels), 
                    replace=False,
                ) for _ in range(batch_size)
            ]

            channels_subset_indices = np.concatenate(channels_subset_idx, axis=0)
            channels_subset_indices = torch.tensor(channels_subset_indices, dtype=torch.long)
            
            channels_subset_indices = channels_subset_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # Shape: [batch_size, num_sampled_channels, H, W]
            masked_img = torch.gather(img, dim=1, index=channels_subset_indices).to(device) 

            # Gather corresponding channel IDs
            active_channel_ids = torch.gather(channel_ids, dim=1, index=channels_subset_indices[:, :, 0, 0]).to(device)

            channel_ids = channel_ids.to(device)
            masked_img = masked_img.to(torch.float32)
            img = img.to(device)

            # output = model(masked_img, active_channel_ids, channel_ids)['output']
            output = model(masked_img, active_channel_ids, channel_ids)['output'][:, :, 3:-4, 3:-4]  # Remove padding
            # output = model(masked_img, active_channel_ids, active_channel_ids)['output'][:, :, 3:-4, 3:-4]  # Remove padding
            # output = model(masked_img, active_channel_ids, active_channel_ids)[0][:, :, 3:-4, 3:-4]  # Remove padding
            mi, logsigma = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)
  
            loss = nll_loss(img, mi, logsigma)
            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()

            if idx in plot_indices:
                uncertainty_img = torch.exp(logsigma)
                unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]
                # unactive_channels = []
                masked_channels_names = '\n'.join([marker_names_map[i.item()] for i in unactive_channels])

                reconstr_img = plot_reconstructs_with_uncertainty(
                    img,
                    mi,
                    uncertainty_img,
                    channel_ids,
                    unactive_channels,
                    markers_names_map=marker_names_map,
                    scale_by_max=True
                )
                panel_idx = ["nsclc-panel-1"]  # Placeholder for panel index, can be modified based on your dataset
                img_path = test_dataloader.dataset.curr_img_path
                run['val/imgs'].append(
                    reconstr_img, 
                    description=f'Resuilting outputs (variance scaled by min-max)  (dataset {panel_idx[0]}, image {img_path[0]}, epoch {epoch})'
                                '\n\nMasked channels: {}'.format(masked_channels_names)
                )
                plt.close('all')
                
    
    val_loss = running_loss / len(test_dataloader)
    run['val/loss'].append(val_loss)
    run['val/mae'].append(running_mae / len(test_dataloader))
    
    return val_loss

if __name__ == '__main__':
    # Load the configuration file
    config_path = sys.argv[1]
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    
    with open("secrets/neptune.yaml", 'r') as f:
        secrets = yaml.load(f)


    device = config['device']
    print(f'Using device: {device}')
    
    prefix = config.get("run_prefix", "").strip()         # empty by default
    suffix = build_run_name_suffix()                               # always unique
    run_name = f"{prefix}_{suffix}" if prefix else suffix

    run = neptune.init_run(
        name=run_name,
        project=secrets['neptune_project'],
        api_token=secrets['neptune_api_token'],
        tags=config['tags'],
    )

    # SIZE = config['input_image_size']
    # print(f"INPUT IMAGE SIZE: {SIZE}")
    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = config['num_workers']

    PANEL_CONFIG = YAML().load(open(config['panel_config']))
    TOKENIZER = YAML().load(open(config['tokenizer_config']))
    print(f"Training on datasets: {PANEL_CONFIG['datasets']}")
    MARKERS_SET = {k for dataset in PANEL_CONFIG['datasets'] for k in PANEL_CONFIG['markers'][dataset]}
    print(f"Markers set: {MARKERS_SET}")
    print(f"Number of markers: {len(MARKERS_SET)}")
    TOKENIZER = {k: v for k, v in zip(MARKERS_SET, range(len(MARKERS_SET)))}
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}


    model_config = {
        'num_channels': len(TOKENIZER),
        'superkernel_config': config['superkernel'],
        'encoder_config': config['encoder'],
        'decoder_config': config['decoder'],
    }


    # start of new code 
    oldConfig = TissuesConfig
    oldConfig.BATCH_SIZE = config['batch_size']
    print(oldConfig)

    train_loaders = []
    test_loaders = []
    for data_dir in oldConfig.DATA_DIRS:
        batch_size = oldConfig.BATCH_SIZE
        if data_dir == IMC_PANEL2_DATA_DIR:
            batch_size = int(batch_size / 1.1)
        image_size = oldConfig.INPUT_OUTPUT_IMAGE_SHAPE[0]

        if data_dir in [IMC_PANEL2_DATA_DIR, IMC_PANEL1_DATA_DIR]:
            # slides_data_module = _create_whole_slides_data_module(data_dir, batch_size, image_size)
            tissues_data_module = _create_tissue_patches_data_module(data_dir, batch_size, image_size)
            # slides_train_loader, slides_test_loader = _setup_data_loaders(slides_data_module)
            tissues_train_loader, tissues_test_loader = _setup_data_loaders(tissues_data_module)
            # train_loaders += [slides_train_loader, tissues_train_loader]
            # test_loaders += [slides_test_loader, tissues_test_loader]
            train_loaders += [tissues_train_loader]
            test_loaders += [tissues_test_loader]
        else:
            tissues_data_module = _create_tissue_patches_data_module(data_dir, batch_size, image_size)
            tissues_train_loader = _setup_data_loader(tissues_data_module)
            train_loaders += [tissues_train_loader]

    train_dataloader = train_loaders[0]
    test_dataloader = test_loaders[0]
    print(f'Training on {len(train_dataloader.dataset)} training samples and {len(test_dataloader.dataset)} test samples')
    # print(f'Batch size: {BATCH_SIZE}, Number of workers: {NUM_WORKERS}')

    # end of new code

    if config["model_type"] == "EquivariantConvnext":
        from multiplex_model.equivariant_modules import EquivariantMultiplexAutoencoder
        model = EquivariantMultiplexAutoencoder(**model_config).to(device)

    elif config["model_type"] == "Convnext":
        model = MultiplexAutoencoder(**model_config).to(device)

    # from src_from_rudy.models.multiplexvit_no_bottleneck_autoencoder_escnn import EscnnMultiplexVitNoBottleneckAutoencoderModule
    # autoencoder_module = EscnnMultiplexVitNoBottleneckAutoencoderModule(
    #     image_size=oldConfig.INPUT_OUTPUT_IMAGE_SHAPE[0],
    #     channels=oldConfig.CHANNELS,
    #     hidden_size=oldConfig.HIDDEN_SIZE,
    #     lr=oldConfig.LR,
    #     debug_image_log_interval=oldConfig.DEBUG_IMAGE_LOG_INTERVAL,
    #     loss_function=oldConfig.LOSS_FUNCTION,
    #     config=config
    # )
    # model = autoencoder_module.autoencoder

    # print(f'Model created with config: {model_config}')
    print(f'Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    print(f'Model: {model}')


    lr = config['lr']
    final_lr = config['final_lr']
    weight_decay = config['weight_decay']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    epochs = config['epochs']
    num_warmup_steps = config['num_warmup_steps']
    num_annealing_steps = config["num_annealing_steps"] #len(train_dataloader) * epochs // gradient_accumulation_steps - num_warmup_steps
    
    # steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    # num_warmup_steps *= steps_per_epoch
    # num_annealing_steps *= steps_per_epoch

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler_with_warmup(optimizer, num_warmup_steps, num_annealing_steps, final_lr=final_lr, type='cosine')

    if 'from_checkpoint' in config and config['from_checkpoint']:
        print(f'Loading model from checkpoint: {config["from_checkpoint"]}')
        checkpoint = torch.load(config['from_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    
    run["slurm/job_id"] = SLURM_JOB_ID
    # run["sys/run_name"] = run_name

    run["parameters"] = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "lr": lr,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "num_warmup_steps": num_warmup_steps,
        "num_annealing_steps": num_annealing_steps,
        # "model_config": stringify_unsupported(model_config),
        "from_checkpoint": config.get('from_checkpoint', None),
    }

    train_masked(
        model, 
        optimizer, 
        scheduler,
        train_dataloader, 
        test_dataloader, 
        device, 
        epochs=epochs, 
        start_epoch=start_epoch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        run=run,
        min_channels_frac=config['min_channels_frac'],
        save_checkpoint_every=config['save_checkpoint_freq'],
        marker_names_map=INV_TOKENIZER,
    )

    run.stop()
