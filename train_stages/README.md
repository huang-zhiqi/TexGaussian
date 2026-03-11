# 两阶段渐进训练方案（PSNR + FID/CLIP 联合监督）

本目录包含两阶段训练脚本，从预训练 PBR 基础模型出发，使用“重建监督 + 语义/分布监督”联合优化。

## 训练流程概览

```
阶段1: FID/CLIP 导向新模块训练
    base UNet 冻结 + FID-safe 参数策略
    重建损失 + CLIP(image/text) + color stats
    ↓ 验证：FID/KID/CLIP_FID/CMMD 下降，PSNR 稳定
阶段2: 低LR微调（可选）
    解冻 base（显式关闭 fid_safe_mode），小学习率短程优化
    继续联合监督并以 FID 选 best ckpt
    ↓ 验证：分布指标进一步改善且 PSNR 不明显退化
```

## 快速开始

```bash
cd train_stages

# 设置可执行权限
chmod +x *.sh

# 第1步：训练所有新增模块
./stage1_new_modules.sh

# 第2步（可选）：低LR微调
./stage2_finetune.sh
```

## 各阶段详情

### 阶段1：FID/CLIP 导向新模块训练（base 冻结）

| 项目 | 说明 |
|------|------|
| 目标 | 在保持重建质量基础上，优先优化分布/语义指标 |
| 时间 | 20 epochs，耗时随 GPU 数量变化（见下方时间预估） |
| 学习率 | 3e-4 |
| 监督 | MSE/material/mask/LPIPS + CLIP(image/text) + color_stats |
| best 选择 | FID |

**新增模块参数分布：**

| 模块 | 参数量 | 说明 |
|------|--------|------|
| GGCA | 1.09M | inner_dim=256, 8 heads, FFN, geometry gate |
| TextAdapter | 394K | 768→256→768 residual MLP |

**检查点：**
- [ ] `eval_fid` 下降（或保持低位）
- [ ] `clip_image_loss / clip_text_loss / color_stats_loss` 稳定下降
- [ ] 重建指标（psnr/albedo_loss/material_loss）不明显崩坏

### 阶段2：FID/CLIP 低LR微调（可选）

| 项目 | 说明 |
|------|------|
| 目标 | 小步解冻优化，追求分布指标进一步提升 |
| 时间 | 8 epochs，约 2-4 小时 |
| 学习率 | 5e-5 |
| 关键开关 | `fid_safe_mode=False`（否则会强制冻结 base） |
| best 选择 | FID |

**检查点：**
- [ ] 与阶段1对比，FID/KID/CLIP_FID/CMMD 改善
- [ ] PSNR 没有明显退化

## GPU 配置

默认使用 4 卡（`GPU_IDS="3,4,5,6"`）。如需切换，修改 `common_config.sh`：

```bash
# 在 common_config.sh 中修改
GPU_IDS="0,1,2,3"
NUM_GPUS=4
```

## 时间预估

| 配置 | 阶段1 | 阶段2 | 总计 |
|------|-------|-------|------|
| 2×3090 | 2-3h | 2-4h | ~4-7h |
| 4×3090 | 1-1.5h | 1-2h | ~2-3.5h |

## 手动指定 Checkpoint

如果自动查找失败，可以手动指定：

```bash
RESUME_CKPT=/path/to/model.safetensors ./stage1_new_modules.sh
```

## 监控指标

```bash
tensorboard --logdir ../experiments
```

关键指标：
- `eval_fid`: 应该下降或保持低位
- `clip_image_loss` / `clip_text_loss` / `color_stats_loss`: 稳定下降
- `psnr`: 保持稳定，不明显崩坏

## 问题排查

### 1. albedo 颜色不对
- 检查 LongCLIP 模型是否正确加载
- 检查 checkpoint 是否正确加载（看 missing/unexpected keys）

### 2. 训练中 OOM
- 减小 `GRAD_ACC`（当前默认为 2）
- 确认 gradient checkpointing 已启用（默认启用）
- 阶段2全模型微调比阶段1消耗更多显存，可适当减小 batch
