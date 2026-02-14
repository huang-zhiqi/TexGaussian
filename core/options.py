import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional, List


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    input_depth: int = 8
    full_depth: int = 4
    in_channels: int = 3
    out_channels: int = 7
    input_feature: str = 'ND'
    # Unet definition
    model_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 8)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True)
    mid_attention: bool = True
    up_attention: Tuple[bool, ...] = (True, True, False, False, False)
    num_heads: int = 16
    context_dim: int = 768
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 128
    # gaussian render size
    output_size: int = 512
    use_material: str = 'True'
    gaussian_loss: str = 'False'
    use_normal_head: str = 'False'
    use_rotation_head: str = 'False'
    use_text: str = 'True'
    use_local_pretrained_ckpt: str = 'False'
    text_description: str = 'Cap3D_automated_Objaverse_full.csv'
    ema_rate: float = 0.999
    radius: float = 1/2
    use_checkpoint: str = 'True'
    lambda_geo_normal: float = 1.0
    lambda_tex_normal: float = 1.0

    ## fit gaussians
    gaussian_list: str = 'pbr_train_list_gaussian.txt'
    mean_path: str = 'statistics/gaussian_mean.pth'
    std_path: str = 'statistics/gaussian_std.pth'

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 30
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    total_num_views: int = 64
    reference_num_views: int = 15
    # num workers
    num_workers: int = 1
    reference_image_mode: str = 'albedo'
    trainlist: str = 'pbr_train_list.txt'
    testlist: str = 'pbr_train_list.txt'
    image_dir: str = 'path_to_image_dir'
    test_image_dir: str = ''
    pointcloud_dir: str = 'path_to_pointcloud_dir'
    gaussian_dir: str = 'path_to_fitted_gaussian'

    ### texture baking
    text_prompt: str = ''
    caption_field: str = 'caption_short'
    use_longclip: str = 'False'
    longclip_model: str = 'third_party/Long-CLIP/checkpoints/longclip-L.pt'
    longclip_context_length: int = 248
    tsv_path: Optional[str] = None
    batch_path: Optional[str] = None
    result_tsv: Optional[str] = None
    max_samples: int = -1  # Maximum number of samples to process (-1 means all)
    texture_cam_radius: float = 4.5
    texture_name: str = 'test'
    save_image: str = 'False'
    num_gpus: int = 1  # Number of GPUs to use for parallel processing
    gpu_ids: str = '0'  # Comma-separated GPU IDs (e.g., '0,1,2,3')
    workers_per_gpu: str = 'auto'  # Workers per GPU: 'auto' or integer (e.g., '2')
    worker_id: int = -1  # Internal: worker ID for multi-GPU mode (-1 means single GPU mode)
    mesh_path: str = '/workspace/wudang_vuc_3dc_afs/wuchenming/bpfs/pbr_obj_cvpr'
    output_dir: str = 'texture_mesh'
    ckpt_path: str = ''

    ### training
    # workspace
    workspace: str = 'workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 8
    # ckpt interval
    ckpt_interval: int = 1
    # image interval
    image_interval: int = 100
    # gradient accumulation
    gradient_accumulation_steps: int = 2
    # training epochs
    num_epochs: int = 1000
    # lpips loss weight
    lambda_lpips: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 4e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = True
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False


# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['objaverse'] = 'the default settings of Objaverse'
config_defaults['objaverse'] = Options()

config_doc['shapenet'] = 'the default settings of shapenet'
config_defaults['shapenet'] = Options(
    use_material = 'False',
    use_text = 'False',
    model_channels=32,
    down_attention = (False, False, False, False, False),
    mid_attention = False,
    up_attention = (False, False, False, False, False),
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
