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
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import nll_loss
from multiplex_model.utils import ClampWithGrad, plot_reconstructs_with_uncertainty, get_scheduler_with_warmup
from multiplex_model.modules import MultiplexAutoencoder

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

    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}')):
            batch_size, num_channels, H, W = img.shape

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

            # Gather corresponding channel IDs
            active_channel_ids = torch.gather(channel_ids, dim=1, index=channels_subset_indices[:, :, 0, 0]).to(device)


            img = img.to(device)
            channel_ids = channel_ids.to(device)
            masked_img = masked_img.to(torch.float32)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(masked_img, active_channel_ids, channel_ids)['output']
                mi, logsigma = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)

                # Apply ClampWithGrad to logsigma for stability
                logsigma = ClampWithGrad.apply(logsigma, -15.0, 15.0)
                loss = nll_loss(img, mi, logsigma)

                # sanity check if loss is finite
                if not torch.isfinite(loss):
                    print(f'Non-finite loss encountered at batch {batch_idx} in epoch {epoch}. Skipping batch.')
                    print(f'Dataset: {panel_idx}')
                    print(f'Image path: {img_path}')
                    print(f'Mi: {mi}, Logsigma: {logsigma}')
                    assert False, "Non-finite loss encountered. Check the model and data."

            scaler.scale(loss / gradient_accumulation_steps).backward()
            # scaler.scale(loss).backward()

            if (batch_idx+1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                run['train/loss'].append(loss.item())
                run['train/lr'].append(scheduler.get_last_lr()[0])
                run['train/µ'].append(mi.mean().item())
                run['train/logvar'].append(logsigma.mean().item())
                run['train/mae'].append(torch.abs(img - mi).mean().item())


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
        num_plots=5, 
        min_channels_frac=0.5
        ):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    plot_indices = np.random.choice(np.arange(len(test_dataloader)), size=num_plots, replace=False)
    plot_indices = set(plot_indices)
    rand_gen = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path) in enumerate(tqdm(test_dataloader, desc=f'Testing epoch {epoch}')):
            batch_size, num_channels, H, W = img.shape
            min_channels = int(np.ceil(num_channels * min_channels_frac))
            
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

            output = model(masked_img, active_channel_ids, channel_ids)['output']
            mi, logsigma = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)
  
            loss = nll_loss(img, mi, logsigma)
            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()

            if idx in plot_indices:
                uncertainty_img = torch.exp(logsigma)
                unactive_channels = [i for i in channel_ids[0] if i not in active_channel_ids[0]]
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


    device = config['device']
    print(f'Using device: {device}')

    SIZE = config['input_image_size']
    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = config['num_workers']

    PANEL_CONFIG = YAML().load(open(config['panel_config']))
    TOKENIZER = YAML().load(open(config['tokenizer_config']))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    train_transform = Compose([
        RandomRotation(
            180, 
            interpolation=InterpolationMode.BILINEAR
        ),
        RandomCrop(SIZE),
    ])

    test_transform = TestCrop(SIZE[0])

    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='train',
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True
    )

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split='test',
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True
    )

    train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
    test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=NUM_WORKERS)

    model_config = {
        'num_channels': len(TOKENIZER),
        'superkernel_config': config['superkernel'],
        'encoder_config': config['encoder'],
        'decoder_config': config['decoder'],
    }

    model = MultiplexAutoencoder(**model_config).to(device)

    lr = config['lr']
    final_lr = config['final_lr']
    weight_decay = config['weight_decay']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    epochs = config['epochs']
    num_warmup_steps = config['num_warmup_steps']
    num_annealing_steps = len(train_dataloader) * epochs // gradient_accumulation_steps - num_warmup_steps

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

    run = neptune.init_run(
        project=config['neptune_project'],
        api_token=config['neptune_api_token'],
        tags=config['tags'],
    )

    run["parameters"] = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "lr": lr,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "num_warmup_steps": num_warmup_steps,
        "num_annealing_steps": num_annealing_steps,
        "model_config": stringify_unsupported(model_config),
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
