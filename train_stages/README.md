# 两阶段渐进训练方案

本目录包含两阶段训练脚本，从预训练 PBR 基础模型出发，训练所有新增创新模块。

## 训练流程概览

```
阶段1: 训练新增模块 (2-3h / 2×3090)
    TextAdapter + GGCA (conv_out入口，enriches ALL features)
    base UNet 冻结，仅训练 ~1.48M 新参数
    ↓ 验证：albedo 颜色正确，loss 稳步下降，PSNR 22-25
阶段2: 全模型微调 (4-5h / 2×3090，可选)
    解冻全部 ~297M 参数，低学习率端到端优化
    ↓ 验证：PSNR 进一步提升，无过拟合
```

## 快速开始

```bash
cd train_stages

# 设置可执行权限
chmod +x *.sh

# 第1步：训练所有新增模块
./stage1_new_modules.sh

# 第2步（可选）：全模型微调
./stage2_finetune.sh
```

## 各阶段详情

### 阶段1：训练新增模块（base 冻结）

| 项目 | 说明 |
|------|------|
| 目标 | 训练 TextAdapter + GGCA |
| 时间 | 20 epochs，约 2-3 小时 (2×3090) |
| 学习率 | 4e-4 |
| 训练参数 | ~1.48M（占 base 的 0.50%） |
| 冻结模块 | 基础 UNet backbone (295M) |

**新增模块参数分布：**

| 模块 | 参数量 | 说明 |
|------|--------|------|
| GGCA | 1.09M | inner_dim=256, 8 heads, FFN, geometry gate |
| TextAdapter | 394K | 768→256→768 residual MLP |

**检查点：**
- [ ] albedo 颜色正确
- [ ] albedo_loss / material_loss 稳步下降
- [ ] PSNR 达到 22-25

### 阶段2：全模型微调（可选）

| 项目 | 说明 |
|------|------|
| 目标 | 端到端微调，进一步提升效果 |
| 时间 | 20 epochs，约 4-5 小时 (2×3090) |
| 学习率 | 5e-5（低学习率，避免破坏预训练） |
| 训练参数 | ~297M（全部参数） |
| 冻结模块 | 无 |

**检查点：**
- [ ] 与阶段1对比，PSNR 有提升
- [ ] 没有过拟合（loss 不应上升）

## GPU 配置

默认使用 2 张 3090。修改 `common_config.sh` 切换到 4 卡：

```bash
# 在 common_config.sh 中修改
GPU_IDS="0,1,2,3"
NUM_GPUS=4
```

## 时间预估

| 配置 | 阶段1 | 阶段2 | 总计 |
|------|-------|-------|------|
| 2×3090 | 2-3h | 4-5h | ~7h |
| 4×3090 | 1.5h | 2.5h | ~4h |

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
- `albedo_loss`: 应该持续下降
- `psnr`: 应该持续上升

## 问题排查

### 1. albedo 颜色不对
- 检查 LongCLIP 模型是否正确加载
- 检查 checkpoint 是否正确加载（看 missing/unexpected keys）

### 2. 训练中 OOM
- 减小 `GRAD_ACC`（当前为 4）
- 确认 gradient checkpointing 已启用（默认启用）
- 阶段2全模型微调比阶段1消耗更多显存，可适当减小 batch
