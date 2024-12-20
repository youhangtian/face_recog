import os 
import sys 
import argparse 

import math
import cv2
import numpy as np 
from tqdm import tqdm

import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 

from accelerate import Accelerator

from src.utils import get_config_from_yaml, get_logger, cosine_scheduler, AverageMeter 
from src.megaface_test import get_mega_dataloader, get_acc 

from src.model import RecogModel
from src.data import get_dataloader

def get_config():
    parser = argparse.ArgumentParser(description='training argument parser')
    parser.add_argument('-c', '--config_file', default='cfg.yaml', type=str, help='config file')
    parser.add_argument('opts', help='modify config options from the command-line', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_config_from_yaml(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze
    return cfg

if __name__ == '__main__':
    cfg = get_config()

    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=2)
    if accelerator.is_main_process:
        os.makedirs(cfg.output, exist_ok=True)
        logger = get_logger(cfg.output, f'log.txt')
        logger.info(f'config: ------\n{cfg}\n------')
        logger.info(f'accelerator: ------\n{accelerator.state}\n------')
    else:
        logger = None

    data_loader = get_dataloader(cfg.model.backbone.input_size, cfg.model.backbone.patch_size, cfg.data)
    if logger: logger.info(f'dataloader len: {len(data_loader)} ------')

    num_classes = len(os.listdir(cfg.data.image_folder))
    model = RecogModel(cfg.model, num_classes)
    if logger: logger.info(f'model: ------\n{model}\n------')

    opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.train.lr, 
        weight_decay=cfg.train.weight_decay
    )
    if logger: logger.info(f'optimizer: ------\n{opt}\n------')

    lr_scheduler = cosine_scheduler(
        cfg.train.lr, 
        cfg.train.lr_end,
        cfg.train.epochs,
        len(data_loader) // accelerator.num_processes,
        warmup_epochs=cfg.train.warmup_epochs
    )

    data_loader = accelerator.prepare(data_loader)
    model = accelerator.prepare(model)
    opt = accelerator.prepare(opt)

    megaface_folders = cfg.data.megaface_face_folders.split(',') if cfg.data.megaface_data_root else []
    megaface_dataloaders = []
    for i in range(len(megaface_folders)):
        megaface_dataloader = get_mega_dataloader(cfg.data.megaface_data_root, 
                                                  megaface_folders[i], 
                                                  cfg.data.batch_size, 
                                                  cfg.model.backbone.input_size)
        megaface_dataloaders.append(megaface_dataloader)
    if logger: logger.info(f'megaface dataloaders: {megaface_folders} ------')

    best_acc = {name: 0.0 for name in megaface_folders[:-1]}

    loss_am = AverageMeter() if logger else None
    writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'tensorboard')) if logger else None

    steps = 0
    for epoch in range(cfg.train.epochs):   
        accelerator.wait_for_everyone()
        if logger: 
            logger.info(f'epoch {epoch} begin ------')
            if epoch % cfg.train.save_epochs == 0:
                save_path = os.path.join(cfg.output, f'model_epoch{epoch}.pth')
                model.module.save_model(save_path)
        
        accelerator.wait_for_everyone()
        for (imgs, labels, masks) in tqdm(data_loader, f'[epoch{epoch}][{accelerator.device}]'):
            for param_group in opt.param_groups:
                param_group['lr'] = lr_scheduler[steps]

            with accelerator.autocast():
                features, attn, loss = model(imgs, labels=labels, masks=masks)

            if not math.isfinite(loss.item()):
                sys.exit(f'loss is {loss.item()}, stopping training')

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 5)

            opt.step()
            opt.zero_grad()

            with torch.no_grad():
                if loss_am: loss_am.update(loss.item())

                if writer:
                    writer.add_scalar(f'train/learning_rate', opt.param_groups[0]['lr'], steps)
                    writer.add_scalar(f'train/loss', loss_am.val, steps)

                    if steps % cfg.image_writer_freq == 0:
                        batch_size = imgs.shape[0]
                        mat = F.linear(features, features)
                        sort_list = mat.flatten().argsort()

                        max_idx = sort_list[-batch_size-1].item()
                        min_idx = sort_list[0].item()
                        max_x, max_y = max_idx // batch_size, max_idx % batch_size 
                        min_x, min_y = min_idx // batch_size, min_idx % batch_size 

                        img_arr = [imgs[idx].cpu().numpy() for idx in [max_x, max_y, min_x, min_y]]
                        for j in range(len(img_arr)):
                            img_arr[j] = img_arr[j].transpose([1, 2, 0])
                            img_arr[j] = img_arr[j] * 127.5 + 127.5 

                        concat_img = np.concatenate(img_arr, axis=1).astype('uint8')
                        concat_img = concat_img.copy()

                        input_size = cfg.model.backbone.input_size 
                        patch_size = cfg.model.backbone.patch_size

                        cv2.putText(concat_img, f'{mat[max_x][max_y].item():.2f}', (input_size[1] - 20, 20), 0, 1, (255, 255, 0))
                        cv2.putText(concat_img, f'{mat[min_x][min_y].item():.2f}', (input_size[1] * 3 - 20, 20), 0, 1, (255, 255, 0))

                        attn = attn.cpu()
                        attn_arr = torch.cat([attn[idx].unsqueeze(0) for idx in [max_x, max_y, min_x, min_y]])
                        num, num_heads, _ = attn_arr.shape  
                        attn_arr = attn_arr.reshape(num, num_heads, input_size[0]//patch_size, input_size[1]//patch_size)
                        attn_arr = F.interpolate(attn_arr, scale_factor=patch_size, mode='nearest').numpy()

                        attn_arr_sum = []
                        for j in range(len(attn_arr)):
                            arr_sum = sum(attn_arr[j][k] for k in range(num_heads))
                            arr_sum = arr_sum / arr_sum.max() * 255 
                            attn_arr_sum.append(arr_sum.astype('uint8'))

                        attn_concat_img = np.concatenate((attn_arr_sum), axis=1)
                        attn_concat_img = np.expand_dims(attn_concat_img, axis=-1)
                        attn_concat_img = np.repeat(attn_concat_img, 3, axis=-1)

                        show_img = np.concatenate([concat_img, attn_concat_img], axis=0)[:,:,::-1]
                        writer.add_image(f'image/heatmap', show_img, steps, dataformats='HWC')

            if (steps % cfg.log_freq == 0) and logger:
                logger.info(f'[epoch{epoch}] steps {steps}, lr {opt.param_groups[0]['lr']:.8f}, loss {loss_am.avg:.4f} ------')

            steps += 1

        accelerator.wait_for_everyone()
        if len(megaface_folders) > 1:
            model.eval()
            with accelerator.autocast():
                acc_arr = get_acc(model, megaface_dataloaders, device=accelerator.device)

            if logger:
                for i in range(len(megaface_folders) - 1):
                    logger.info(f'[epoch{epoch}][megaface_test][{megaface_folders[i]}] acc {acc_arr[i]*100:.2f}% ------')
                    writer.add_scalar(f'train/test_{megaface_folders[i]}', acc_arr[i], epoch)

                    if acc_arr[i] > best_acc[megaface_folders[i]]:
                        best_acc[megaface_folders[i]] = acc_arr[i] 
                        save_path = os.path.join(cfg.output, f'backbone_{megaface_folders[i]}.pth')
                        model.module.save_backbone(save_path)
            model.train()

    accelerator.wait_for_everyone()
    if writer: writer.close()
    if logger:
        model.eval()
        save_path = os.path.join(cfg.output, 'model_final.pth')
        model.module.save_model(save_path)
        logger.info(f'training done, best acc arr: {best_acc} ------')
