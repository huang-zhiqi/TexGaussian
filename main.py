import tyro
import time
import random
import os
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
    opt.use_text = str2bool(opt.use_text)
    opt.use_longclip = str2bool(opt.use_longclip)
    opt.use_local_pretrained_ckpt = str2bool(opt.use_local_pretrained_ckpt)

    current_time = get_time()

    experiment_name = f"{current_time}_lr_{opt.lr}_num_views_{opt.num_views}"

    opt.workspace = os.path.join(opt.workspace, experiment_name)

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

    train_dataset = Dataset(opt, training=True)
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
        (opt.use_normal_head or opt.use_rotation_head)
        and (hasattr(model, "normal_head") or hasattr(model, "rotation_head"))
    )
    if use_head_lr_split:
        head_params = []
        if hasattr(model, "normal_head"):
            head_params += list(model.normal_head.parameters())
        if hasattr(model, "rotation_head"):
            head_params += list(model.rotation_head.parameters())
        head_param_ids = {id(p) for p in head_params}
        base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
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

    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = min(0.3, max(0.01, 3000.0 / max(1, total_steps)))
    if use_head_lr_split:
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

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
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

                    pred_images = out['images_pred'] # [B, 3, output_size, output_size]

                    pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_{uid[0]}.jpg', pred_images)
                    else:
                        kiui.write_image(f'{opt.pred_image_dir}/{epoch}_{i}.jpg', pred_images)

                    if opt.use_material:
                        rough_pred = out['rough_images_pred'].detach().cpu().numpy()  # [B, V, 1, H, W]
                        rough_pred = np.repeat(rough_pred, 3, axis=2)
                        rough_pred = rough_pred.transpose(0, 3, 1, 4, 2).reshape(-1, rough_pred.shape[1] * rough_pred.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_rough_{uid[0]}.jpg', rough_pred)
                        else:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_rough_{i}.jpg', rough_pred)

                        metal_pred = out['metallic_images_pred'].detach().cpu().numpy()  # [B, V, 1, H, W]
                        metal_pred = np.repeat(metal_pred, 3, axis=2)
                        metal_pred = metal_pred.transpose(0, 3, 1, 4, 2).reshape(-1, metal_pred.shape[1] * metal_pred.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_metal_{uid[0]}.jpg', metal_pred)
                        else:
                            kiui.write_image(f'{opt.pred_image_dir}/{epoch}_metal_{i}.jpg', metal_pred)

                    gt_images = data['images_output']

                    gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    if opt.batch_size == 1:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_{uid[0]}.jpg', gt_images)
                    else:
                        kiui.write_image(f'{opt.gt_image_dir}/{epoch}_{i}.jpg', gt_images)

                    if opt.use_material:
                        rough_gt_images = data['rough_images_output'].detach().cpu().numpy()
                        rough_gt_images = np.repeat(rough_gt_images, 3, axis=2)
                        rough_gt_images = rough_gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, rough_gt_images.shape[1] * rough_gt_images.shape[3], 3)
                        if opt.batch_size == 1:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_rough_{uid[0]}.jpg', rough_gt_images)
                        else:
                            kiui.write_image(f'{opt.gt_image_dir}/{epoch}_rough_{i}.jpg', rough_gt_images)

                        metal_gt_images = data['metallic_images_output'].detach().cpu().numpy()
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

                    gt_images = data['images_output']

                    gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_{uid[0]}.jpg', gt_images)

                    pred_images = out['images_pred']
                    pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_{uid[0]}.jpg', pred_images)

                    if opt.use_material:
                        rough_gt = data['rough_images_output'].detach().cpu().numpy()  # [B, V, 1, H, W]
                        rough_gt = np.repeat(rough_gt, 3, axis=2)
                        rough_gt = rough_gt.transpose(0, 3, 1, 4, 2).reshape(-1, rough_gt.shape[1] * rough_gt.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_rough_{uid[0]}.jpg', rough_gt)

                        rough_pred = out['rough_images_pred'].detach().cpu().numpy()  # [B, V, 1, H, W]
                        rough_pred = np.repeat(rough_pred, 3, axis=2)
                        rough_pred = rough_pred.transpose(0, 3, 1, 4, 2).reshape(-1, rough_pred.shape[1] * rough_pred.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_rough_{uid[0]}.jpg', rough_pred)

                        metal_gt = data['metallic_images_output'].detach().cpu().numpy()
                        metal_gt = np.repeat(metal_gt, 3, axis=2)
                        metal_gt = metal_gt.transpose(0, 3, 1, 4, 2).reshape(-1, metal_gt.shape[1] * metal_gt.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_gt_images/{epoch}_metal_{uid[0]}.jpg', metal_gt)

                        metal_pred = out['metallic_images_pred'].detach().cpu().numpy()
                        metal_pred = np.repeat(metal_pred, 3, axis=2)
                        metal_pred = metal_pred.transpose(0, 3, 1, 4, 2).reshape(-1, metal_pred.shape[1] * metal_pred.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images/{epoch}_metal_{uid[0]}.jpg', metal_pred)

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
            

    if accelerator.is_main_process:
        writer.close()

if __name__ == "__main__":
    main()
