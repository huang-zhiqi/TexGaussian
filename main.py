import tyro
import time
import random
import os
import gc
import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from core.utils import seed_everything
from core.options import AllConfigs
from core.regression_models import TexGaussian
from accelerate import Accelerator
from safetensors.torch import load_file
from core.dataset import TexGaussianDataset as Dataset
from core.dataset import collate_func

import kiui
import pytz
from datetime import datetime
import os


def save_training_state(accelerator, optimizer, scheduler, epoch, max_eval_psnr, save_path):
    """
    保存完整的训练状态，用于恢复中断的训练。
    
    保存内容：
    - epoch: 当前完成的 epoch 数
    - max_eval_psnr: 最佳评估 PSNR
    - optimizer_state_dict: 优化器状态 (momentum, adam states等)
    - scheduler_state_dict: 学习率调度器状态
    """
    if not accelerator.is_main_process:
        return
    
    state = {
        'epoch': epoch,
        'max_eval_psnr': max_eval_psnr,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    os.makedirs(save_path, exist_ok=True)
    state_file = os.path.join(save_path, 'training_state.pt')
    torch.save(state, state_file)
    accelerator.print(f"[INFO] Saved training state to {state_file} (epoch={epoch}, max_psnr={max_eval_psnr:.4f})")


def load_training_state(accelerator, optimizer, scheduler, state_path):
    """
    恢复训练状态。
    
    返回：
    - start_epoch: 从哪个 epoch 开始继续训练
    - max_eval_psnr: 恢复的最佳评估 PSNR
    """
    if not os.path.exists(state_path):
        accelerator.print(f"[WARN] Training state file not found: {state_path}")
        return 0, 0.0
    
    state = torch.load(state_path, map_location='cpu')
    
    # 恢复 optimizer 状态
    try:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        accelerator.print(f"[INFO] Restored optimizer state")
    except Exception as e:
        accelerator.print(f"[WARN] Failed to restore optimizer state: {e}")
    
    # 恢复 scheduler 状态
    try:
        scheduler.load_state_dict(state['scheduler_state_dict'])
        accelerator.print(f"[INFO] Restored scheduler state")
    except Exception as e:
        accelerator.print(f"[WARN] Failed to restore scheduler state: {e}")
    
    start_epoch = state.get('epoch', 0) + 1  # 从下一个 epoch 开始
    max_eval_psnr = state.get('max_eval_psnr', 0.0)
    
    accelerator.print(f"[INFO] Resuming training from epoch {start_epoch}, max_eval_psnr={max_eval_psnr:.4f}")
    
    return start_epoch, max_eval_psnr


def get_time():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)
    timestr = beijing_time.strftime("%Y.%m.%d-%H:%M:%S")
    return timestr

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    opt = tyro.cli(AllConfigs)

    opt.use_checkpoint = str2bool(opt.use_checkpoint)
    opt.use_material = str2bool(opt.use_material)
    opt.gaussian_loss = str2bool(opt.gaussian_loss)
    opt.use_normal_head = str2bool(opt.use_normal_head)
    opt.use_rotation_head = str2bool(opt.use_rotation_head)
    opt.use_ggca = str2bool(opt.use_ggca)
    opt.use_text_adapter = str2bool(opt.use_text_adapter)
    opt.freeze_base = str2bool(opt.freeze_base)
    opt.use_text = str2bool(opt.use_text)
    opt.use_longclip = str2bool(opt.use_longclip)
    opt.use_local_pretrained_ckpt = str2bool(opt.use_local_pretrained_ckpt)

    # 决定 workspace 路径
    # 如果是恢复训练（指定了 resume_training_state），使用原来的 workspace
    # 否则创建新的带时间戳的文件夹
    #
    # 多 GPU 注意：accelerate launch 会为每个 GPU 启动独立进程，
    # 只有 rank 0 生成时间戳并创建目录，其他进程等待后读取。
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if opt.resume_training_state is not None and os.path.exists(opt.resume_training_state):
        # training_state.pt 在 best_ckpt/ 目录下
        # workspace 结构: experiments/texverse_stage2_xxx/2026.02.17-xxx/best_ckpt/training_state.pt
        # 需要往上两级找到实际的 experiment workspace
        state_dir = os.path.dirname(opt.resume_training_state)  # best_ckpt/
        opt.workspace = os.path.dirname(state_dir)               # 2026.02.17-xxx/
        print(f"[INFO] Resuming into existing workspace: {opt.workspace}")
    else:
        # Rank 0 生成时间戳，写入临时文件；其他 rank 等待并读取
        workspace_base = opt.workspace
        stamp_file = os.path.join(workspace_base, ".workspace_stamp")

        if local_rank == 0:
            current_time = get_time()
            experiment_name = f"{current_time}_lr_{opt.lr}_num_views_{opt.num_views}"
            opt.workspace = os.path.join(workspace_base, experiment_name)
            os.makedirs(opt.workspace, exist_ok=True)
            # 写入标记文件供其他 rank 读取
            with open(stamp_file, "w") as f:
                f.write(opt.workspace)
        else:
            # 等待 rank 0 创建标记文件（最多 60 秒）
            import time as _time
            for _ in range(600):
                if os.path.exists(stamp_file):
                    break
                _time.sleep(0.1)
            with open(stamp_file, "r") as f:
                opt.workspace = f.read().strip()

    os.makedirs(opt.workspace, exist_ok = True)

    opt.gt_image_dir = os.path.join(opt.workspace, 'gt_images')
    opt.pred_image_dir = os.path.join(opt.workspace, 'pred_images')

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    device = accelerator.device

    # model
    model = TexGaussian(opt, device)

    if accelerator.is_main_process:

        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'Total Params:{trainable_num / 1e6}M')


    # resume
    if opt.resume is not None:
        print(opt.resume)
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')

        accelerator.print('Start loading checkpoint')
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]

        model_state = model.state_dict()
        filtered_ckpt = {}
        skipped_text = []
        skipped_shape = []
        skipped_nontensor = 0
        unexpected_keys = []

        skip_text_encoder = opt.use_text and opt.use_longclip

        for k, v in ckpt.items():
            if not torch.is_tensor(v):
                skipped_nontensor += 1
                continue

            key = k
            if key not in model_state and key.startswith("module."):
                key = key[len("module."):]

            if key not in model_state:
                unexpected_keys.append(k)
                continue

            if skip_text_encoder and key.startswith("text_encoder."):
                skipped_text.append(key)
                continue

            if model_state[key].shape != v.shape:
                skipped_shape.append((key, tuple(v.shape), tuple(model_state[key].shape)))
                continue

            filtered_ckpt[key] = v

        incompatible = model.load_state_dict(filtered_ckpt, strict=False)
        if incompatible.missing_keys:
            accelerator.print(f"[WARN] missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            accelerator.print(f"[WARN] unexpected keys: {len(incompatible.unexpected_keys)}")
        if unexpected_keys:
            accelerator.print(f"[WARN] checkpoint keys not in model: {len(unexpected_keys)}")
        if skipped_text:
            accelerator.print(f"[INFO] skipped text encoder keys for LongCLIP: {len(skipped_text)}")
        if skipped_shape:
            accelerator.print(f"[WARN] skipped shape-mismatch keys: {len(skipped_shape)}")
            for name, src_shape, dst_shape in skipped_shape[:5]:
                accelerator.print(f"  - {name}: ckpt{src_shape} vs model{dst_shape}")
        if skipped_nontensor:
            accelerator.print(f"[INFO] skipped non-tensor checkpoint entries: {skipped_nontensor}")
        accelerator.print(f"[INFO] loaded checkpoint tensors: {len(filtered_ckpt)}")
        accelerator.print("Loaded base PBR weights. 'normal_head' (and 'rotation_head') weights were initialized freshly.")
        
        # CRITICAL: Sync ema_model with model after loading checkpoint
        # This ensures ema_model starts from the same pretrained weights as model
        model.reset_parameters()  # This copies model weights to ema_model
        accelerator.print("[INFO] Synced ema_model with loaded model weights")

    train_dataset = Dataset(opt, training=True)

    # Sync renderer FOV after dataset auto-detection may have updated opt.fovy
    model.gs.refresh_fov()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        collate_fn=collate_func,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_func,
        pin_memory=True,
        drop_last=False,
    )

    if len(train_dataloader) == 0:
        raise RuntimeError(
            "Training dataloader is empty. Check train.tsv, image_dir, pointcloud_dir, "
            "and reduce batch_size or disable drop_last."
        )
    if len(test_dataloader) == 0:
        raise RuntimeError(
            "Test dataloader is empty. Check test.tsv, image_dir, and pointcloud_dir."
        )

    # optimizer
    use_head_lr_split = (
        (opt.use_normal_head or opt.use_rotation_head or opt.use_ggca or opt.use_text_adapter)
        and (hasattr(model, "normal_head") or hasattr(model, "rotation_head") or 
             (hasattr(model, "model") and hasattr(model.model, "ggca") and model.model.ggca is not None) or
             (hasattr(model, "text_adapter") and model.text_adapter is not None))
    )
    if use_head_lr_split:
        head_params = []
        if hasattr(model, "normal_head"):
            head_params += list(model.normal_head.parameters())
        if hasattr(model, "rotation_head"):
            head_params += list(model.rotation_head.parameters())
        # Include GGCA parameters
        if hasattr(model, "model") and hasattr(model.model, "ggca") and model.model.ggca is not None:
            head_params += list(model.model.ggca.parameters())
        # Include Text Adapter parameters
        if hasattr(model, "text_adapter") and model.text_adapter is not None:
            head_params += list(model.text_adapter.parameters())
        head_param_ids = {id(p) for p in head_params}
        base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
        
        # Freeze base params if freeze_base is enabled
        if opt.freeze_base:
            for p in base_params:
                p.requires_grad = False
            accelerator.print(f"[INFO] Freezing {len(base_params)} base parameters, only training {len(head_params)} head/GGCA/adapter parameters")
            optimizer = torch.optim.AdamW(
                head_params,
                lr=opt.lr,
                weight_decay=0.05,
                betas=(0.9, 0.95),
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": opt.lr * 0.1},
                    {"params": head_params, "lr": opt.lr},
                ],
                weight_decay=0.05,
                betas=(0.9, 0.95),
            )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad == True],
            lr=opt.lr,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )

    # Compute actual optimizer steps accounting for DDP sharding and grad accumulation.
    # Pre-prepare len(train_dataloader) = total_samples / batch_size.
    # After prepare: sharded by num_processes, stepped every grad_acc iterations.
    steps_per_epoch = len(train_dataloader) // (accelerator.num_processes * opt.gradient_accumulation_steps)
    total_steps = opt.num_epochs * max(1, steps_per_epoch)
    accelerator.print(f"[INFO] LR schedule: {steps_per_epoch} optimizer steps/epoch, {total_steps} total steps")
    pct_start = min(0.3, max(0.01, 3000.0 / max(1, total_steps)))
    if use_head_lr_split and not opt.freeze_base:
        # Only use different LRs when not freezing base params
        max_lr = [opt.lr * 0.1, opt.lr]
    else:
        max_lr = opt.lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=max(1, total_steps),
        pct_start=pct_start,
    )

    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(   # prepare函数会自动把模型、优化器、训练集和测试集都加载到gpu上，不用手动移动了。
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir = opt.workspace)

        file_name = os.path.join(opt.workspace, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(opt.__dict__.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        os.system(f'cp main.py {opt.workspace}')
        os.system(f'cp -r core {opt.workspace}')

    iter_start_time = time.time()

    max_eval_psnr = 0
    start_epoch = 0

    # 恢复训练状态（如果指定）
    if opt.resume_training_state is not None:
        start_epoch, max_eval_psnr = load_training_state(
            accelerator, optimizer, scheduler, opt.resume_training_state
        )
        accelerator.print(f"[INFO] Will train from epoch {start_epoch} to {opt.num_epochs-1}")

    # loop
    for epoch in range(start_epoch, opt.num_epochs):
        # train
        model.train()
        # Set current epoch for normal loss warmup scheduling
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.current_epoch = epoch
        total_loss = 0
        total_psnr = 0
        if opt.use_material:
            rough_total_psnr = 0
            metallic_total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                out = model(data)
                loss = out['loss']
                psnr = out['psnr']
                gaussian_loss = out['gaussian_loss']
                albedo_loss = out['albedo_loss']
                mask_loss = out['mask_loss']
                lpips_loss = out['lpips_loss']
                normal_geo_loss = out.get('normal_geo_loss', 0.0)
                normal_tex_loss = out.get('normal_tex_loss', 0.0)

                if opt.use_material:
                    rough_loss = out['roughness_loss']
                    metallic_loss = out['metallic_loss']
                    rough_psnr = out['roughness_psnr']
                    metallic_psnr = out['metallic_psnr']

                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()
                accelerator.unwrap_model(model).update_EMA()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

                if opt.use_material:
                    rough_total_psnr += rough_psnr.detach()
                    metallic_total_psnr += metallic_psnr.detach()

                torch.cuda.empty_cache()

            if accelerator.is_main_process:
                # logging
                if i % 10 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    t = (time.time() - iter_start_time)

#                     print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G \
# lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f} \
# loss: {loss.item():.6f} time: {t:.3f}")

                    if opt.use_material:
                        print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G \
lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f} loss: {loss.item():.6f} \
gaussian_loss: {gaussian_loss:.6f} albedo_loss: {albedo_loss:.6f} roughness_loss: {rough_loss:.6f} \
metallic_loss: {metallic_loss:.6f} mask_loss: {mask_loss:.6f} lpips_loss: {lpips_loss:.6f} \
normal_geo_loss: {normal_geo_loss:.6f} normal_tex_loss: {normal_tex_loss:.6f} time: {t:.3f}")

                    else:
                        print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G \
lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f} \
loss: {loss.item():.6f} gaussian_loss: {gaussian_loss:.6f} albedo_loss: {albedo_loss:.6f} \
mask_loss: {mask_loss:.6f} lpips_loss: {lpips_loss:.6f} \
normal_geo_loss: {normal_geo_loss:.6f} normal_tex_loss: {normal_tex_loss:.6f} time: {t:.3f}")

                    # writer.add_scalar('loss', loss, epoch * len(train_dataloader) + i)

                    writer.add_scalar('loss', out['loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('gaussian_loss', out['gaussian_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('albedo_loss', out['albedo_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('mask_loss', out['mask_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('lpips_loss', out['lpips_loss'], epoch * len(train_dataloader) + i)
                    writer.add_scalar('psnr', out['psnr'], epoch * len(train_dataloader) + i)
                    if opt.use_normal_head or opt.use_rotation_head:
                        writer.add_scalar('normal_geo_loss', out['normal_geo_loss'], epoch * len(train_dataloader) + i)
                        writer.add_scalar('normal_tex_loss', out['normal_tex_loss'], epoch * len(train_dataloader) + i)

                    if opt.use_material:
                        writer.add_scalar('roughness_loss', out['roughness_loss'], epoch * len(train_dataloader) + i)
                        writer.add_scalar('metallic_loss', out['metallic_loss'], epoch * len(train_dataloader) + i)
                        writer.add_scalar('roughness_psnr', out['roughness_psnr'], epoch * len(train_dataloader) + i)
                        writer.add_scalar('metallic_psnr', out['metallic_psnr'], epoch * len(train_dataloader) + i)

                # save log images
                if i % opt.image_interval == 0:
                    # pred_images = model.module.sample_and_render(data)

                    uid = data['uid']
                    train_gt_masks = data['masks_output']  # [B, V, 1, H, W]

                    pred_images = out['images_pred'] # [B, V, 3, output_size, output_size]

                    pred_images = (pred_images * train_gt_masks).detach().cpu().numpy() # mask pred to black bg
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_{uid[0]}.jpg', pred_images)
                    else:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_{i}.jpg', pred_images)

                    if opt.use_material:
                        rough_pred = (out['rough_images_pred'] * train_gt_masks).detach().cpu().numpy()  # [B, V, 1, H, W]
                        rough_pred = np.repeat(rough_pred, 3, axis=2)
                        rough_pred = rough_pred.transpose(0, 3, 1, 4, 2).reshape(-1, rough_pred.shape[1] * rough_pred.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_rough_{uid[0]}.jpg', rough_pred)
                        else:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_rough_{i}.jpg', rough_pred)

                        metal_pred = (out['metallic_images_pred'] * train_gt_masks).detach().cpu().numpy()  # [B, V, 1, H, W]
                        metal_pred = np.repeat(metal_pred, 3, axis=2)
                        metal_pred = metal_pred.transpose(0, 3, 1, 4, 2).reshape(-1, metal_pred.shape[1] * metal_pred.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_metal_{uid[0]}.jpg', metal_pred)
                        else:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_metal_{i}.jpg', metal_pred)

                    # Save normal predictions if available
                    if 'normal_images_pred' in out:
                        normal_pred_vis = out['normal_images_pred'].detach().cpu().numpy()  # [B, V, 3, H, W]
                        normal_pred_vis = normal_pred_vis.transpose(0, 3, 1, 4, 2).reshape(-1, normal_pred_vis.shape[1] * normal_pred_vis.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_normal_{uid[0]}.jpg', normal_pred_vis)
                        else:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_normal_{i}.jpg', normal_pred_vis)

                        normal_gt_vis = out['normal_images_gt'].detach().cpu().numpy()  # [B, V, 3, H, W]
                        normal_gt_vis = normal_gt_vis.transpose(0, 3, 1, 4, 2).reshape(-1, normal_gt_vis.shape[1] * normal_gt_vis.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_normal_{uid[0]}.jpg', normal_gt_vis)
                        else:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_normal_{i}.jpg', normal_gt_vis)

                    # Save pred alpha (mask) and GT mask for visual comparison
                    pred_mask_vis = out['alphas_pred'].detach().cpu().numpy()  # [B, V, 1, H, W]
                    pred_mask_vis = np.repeat(pred_mask_vis, 3, axis=2)  # → [B, V, 3, H, W]
                    pred_mask_vis = pred_mask_vis.transpose(0, 3, 1, 4, 2).reshape(-1, pred_mask_vis.shape[1] * pred_mask_vis.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_mask_{uid[0]}.jpg', pred_mask_vis)
                    else:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_mask_{i}.jpg', pred_mask_vis)

                    gt_mask_vis = data['masks_output'].detach().cpu().numpy()  # [B, V, 1, H, W]
                    gt_mask_vis = np.repeat(gt_mask_vis, 3, axis=2)
                    gt_mask_vis = gt_mask_vis.transpose(0, 3, 1, 4, 2).reshape(-1, gt_mask_vis.shape[1] * gt_mask_vis.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_mask_{uid[0]}.jpg', gt_mask_vis)
                    else:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_mask_{i}.jpg', gt_mask_vis)

                    gt_images = data['images_output'] * data['masks_output']  # mask to black bg

                    gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_{uid[0]}.jpg', gt_images)
                    else:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_{i}.jpg', gt_images)

                    if opt.use_material:
                        rough_gt_images = (data['rough_images_output'] * data['masks_output']).detach().cpu().numpy()
                        rough_gt_images = np.repeat(rough_gt_images, 3, axis=2)
                        rough_gt_images = rough_gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, rough_gt_images.shape[1] * rough_gt_images.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_rough_{uid[0]}.jpg', rough_gt_images)
                        else:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_rough_{i}.jpg', rough_gt_images)

                        metal_gt_images = (data['metallic_images_output'] * data['masks_output']).detach().cpu().numpy()
                        metal_gt_images = np.repeat(metal_gt_images, 3, axis=2)
                        metal_gt_images = metal_gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, metal_gt_images.shape[1] * metal_gt_images.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_metal_{uid[0]}.jpg', metal_gt_images)
                        else:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_metal_{i}.jpg', metal_gt_images)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if opt.use_material:
            rough_total_psnr = accelerator.gather_for_metrics(rough_total_psnr).mean()
            metallic_total_psnr = accelerator.gather_for_metrics(metallic_total_psnr).mean()
        
        # 清理显存碎片（每个 epoch 结束后）
        torch.cuda.empty_cache()
        gc.collect()
        
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            if opt.use_material:
                rough_total_psnr /= len(train_dataloader)
                metallic_total_psnr /= len(train_dataloader)
                accelerator.print(
                    f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} "
                    f"rough_psnr: {rough_total_psnr.item():.4f} metallic_psnr: {metallic_total_psnr.item():.4f}"
                )
            else:
                accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")

        # checkpoint
        if epoch % opt.ckpt_interval == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, opt.workspace)

        total_psnr = 0
        if opt.use_material:
            rough_total_psnr = 0
            metallic_total_psnr = 0

        model.eval()

        # eval
        with torch.no_grad():

            for i, data in enumerate(test_dataloader):

                out = model(data, ema = True)

                psnr = out['psnr']
                total_psnr += psnr.detach()

                if opt.use_material:
                    rough_psnr = out['roughness_psnr']
                    metallic_psnr = out['metallic_psnr']
                    rough_total_psnr += rough_psnr.detach()
                    metallic_total_psnr += metallic_psnr.detach()

                # save some images
                if accelerator.is_main_process:
                    uid = data['uid']
                    gt_masks = data['masks_output']  # [B, V, 1, H, W]

                    # Mask GT to black background (consistent with pred's bg_color=zeros)
                    gt_images = data['images_output'] * gt_masks

                    gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_{uid[0]}.jpg', gt_images)

                    pred_images = out['images_pred']
                    pred_images = (pred_images * gt_masks).detach().cpu().numpy() # mask pred to black bg too
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_{uid[0]}.jpg', pred_images)

                    # Save pred alpha (mask) and GT mask for visual comparison
                    pred_mask_vis = out['alphas_pred'].detach().cpu().numpy()  # [B, V, 1, H, W]
                    pred_mask_vis = np.repeat(pred_mask_vis, 3, axis=2)
                    pred_mask_vis = pred_mask_vis.transpose(0, 3, 1, 4, 2).reshape(-1, pred_mask_vis.shape[1] * pred_mask_vis.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_mask_{uid[0]}.jpg', pred_mask_vis)

                    gt_mask_vis = gt_masks.detach().cpu().numpy()  # [B, V, 1, H, W]
                    gt_mask_vis = np.repeat(gt_mask_vis, 3, axis=2)
                    gt_mask_vis = gt_mask_vis.transpose(0, 3, 1, 4, 2).reshape(-1, gt_mask_vis.shape[1] * gt_mask_vis.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_mask_{uid[0]}.jpg', gt_mask_vis)

                    if opt.use_material:
                        # Mask material GT to black bg (consistent with material renderer)
                        rough_gt = (data['rough_images_output'] * gt_masks).detach().cpu().numpy()  # [B, V, 1, H, W]
                        rough_gt = np.repeat(rough_gt, 3, axis=2)
                        rough_gt = rough_gt.transpose(0, 3, 1, 4, 2).reshape(-1, rough_gt.shape[1] * rough_gt.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_rough_{uid[0]}.jpg', rough_gt)

                        rough_pred = (out['rough_images_pred'] * gt_masks).detach().cpu().numpy()  # mask pred holes
                        rough_pred = np.repeat(rough_pred, 3, axis=2)
                        rough_pred = rough_pred.transpose(0, 3, 1, 4, 2).reshape(-1, rough_pred.shape[1] * rough_pred.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_rough_{uid[0]}.jpg', rough_pred)

                        metal_gt = (data['metallic_images_output'] * gt_masks).detach().cpu().numpy()
                        metal_gt = np.repeat(metal_gt, 3, axis=2)
                        metal_gt = metal_gt.transpose(0, 3, 1, 4, 2).reshape(-1, metal_gt.shape[1] * metal_gt.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_metal_{uid[0]}.jpg', metal_gt)

                        metal_pred = (out['metallic_images_pred'] * gt_masks).detach().cpu().numpy()  # mask pred holes
                        metal_pred = np.repeat(metal_pred, 3, axis=2)
                        metal_pred = metal_pred.transpose(0, 3, 1, 4, 2).reshape(-1, metal_pred.shape[1] * metal_pred.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_metal_{uid[0]}.jpg', metal_pred)

                    # Save normal predictions if available
                    if 'normal_images_pred' in out:
                        normal_pred_vis = out['normal_images_pred'].detach().cpu().numpy()
                        normal_pred_vis = normal_pred_vis.transpose(0, 3, 1, 4, 2).reshape(-1, normal_pred_vis.shape[1] * normal_pred_vis.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_normal_{uid[0]}.jpg', normal_pred_vis)

                        normal_gt_vis = out['normal_images_gt'].detach().cpu().numpy()
                        normal_gt_vis = normal_gt_vis.transpose(0, 3, 1, 4, 2).reshape(-1, normal_gt_vis.shape[1] * normal_gt_vis.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_normal_{uid[0]}.jpg', normal_gt_vis)

                torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if opt.use_material:
            rough_total_psnr = accelerator.gather_for_metrics(rough_total_psnr).mean()
            metallic_total_psnr = accelerator.gather_for_metrics(metallic_total_psnr).mean()
        if accelerator.is_main_process:
            total_psnr /= len(test_dataloader)
            if opt.use_material:
                rough_total_psnr /= len(test_dataloader)
                metallic_total_psnr /= len(test_dataloader)
                accelerator.print(
                    f"[eval] epoch: {epoch} psnr: {total_psnr:.4f} "
                    f"rough_psnr: {rough_total_psnr:.4f} metallic_psnr: {metallic_total_psnr:.4f}"
                )
            else:
                accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f}")

        if total_psnr > max_eval_psnr:
            max_eval_psnr = total_psnr
            accelerator.wait_for_everyone()
            save_path = f'{opt.workspace}/best_ckpt'
            accelerator.save_model(model, save_path)
            # 同时保存训练状态
            save_training_state(accelerator, optimizer, scheduler, epoch, max_eval_psnr, save_path)
        
        # Epoch 结束时的内存清理
        torch.cuda.empty_cache()
        gc.collect()
        accelerator.print(f"[INFO] Epoch {epoch} completed. Memory cleaned.")
            

    if accelerator.is_main_process:
        writer.close()
        # 清理 workspace 同步标记文件
        stamp_file = os.path.join(os.path.dirname(opt.workspace), ".workspace_stamp")
        if os.path.exists(stamp_file):
            os.remove(stamp_file)

if __name__ == "__main__":
    main()
