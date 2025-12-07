import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import re
import time
import torch
import torchvision
import lightning as pl
from torch.utils import tensorboard

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from lightning import seed_everything
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
# from pytorch_lightning.utilities.distributed import rank_zero_only
# from pytorch_lightning.utilities import rank_zero_info
from lightning.fabric.utilities.rank_zero import (  
    rank_zero_info,
    rank_zero_only,
)

# from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
# from ldm.data.dataset import Dataset3D_diffusionGAT_infer


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="temp",
        nargs="?",
        help="postfix for logdir",
    )
    # parser.add_argument(
    #     "-b",
    #     "--base",
    #     nargs="*",
    #     metavar="base_config.yaml",
    #     help="paths to base configs. Loaded from left-to-right. "
    #          "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    #     default=list(),
    # )
    parser.add_argument(
        "-b",
        "--base",
        default="",
        help="paths to base configs"
    )
    parser.add_argument(
        "--CT_root",
        default="",
        help="Path of the original CT volume.",
    )
    parser.add_argument(
        "--resized_CT_root",
        default="",
        help="Path of the resize CT volume.",
    )
    parser.add_argument(
        "--topkMIP_CT_root",
        default="",
        help="topk MIP of the resized CT volume."
    )
    parser.add_argument(
        "--batch_topkMIP_CT_root",
        default="",
        help="topk MIP batches of the resized CT volume."
    )
    parser.add_argument(
        "--rotMat_root",
        default="",
        help="Rotation matrix for different views."
    )
    parser.add_argument(
        "--systemMatrix_root",
        default="",
        help="system matrix for forward and backward projection."
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=98,
        help="Percentile for vessel binarization based on optimized FBP recons. Recommend range [95,98]. Balance the trade-off between over- and under-segmentation."
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

def my_collate_infer(batch):
    proj_ctmip = torch.cat([item[0] for item in batch],axis=0) 
    # proj_vessel = torch.cat([item[1] for item in batch],axis=0) 
    path = [item[2] for item in batch]
    return proj_ctmip, None, path


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, test=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"],
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            drop_last=False,
                            collate_fn=my_collate_infer,
                            pin_memory=False,
                            shuffle=False,
                            prefetch_factor=4,
                            persistent_workers=True
                            )


class SetupCallback(Callback):
    def __init__(self, resume, name, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.name = name
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # if "callbacks" in self.lightning_config:
            #     if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
            #         os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.name)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.name)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        self.start_time = time.time()

    # def on_train_epoch_end(self, trainer, pl_module, outputs):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(0)
        max_memory = torch.cuda.max_memory_allocated(0) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    # init and save configs
    # configs = [OmegaConf.load(cfg) for cfg in opt.base]
    configs = OmegaConf.load(opt.base)
    cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli) # all defined parameters in config.yaml
    config = OmegaConf.merge(configs, cli) # all defined parameters in config.yaml


    print('Data preparation...')
    ####################################################################################################
    from data_prepare import CT_resize, projections, cat_projections_batch

    CT_resize(root_path=opt.CT_root, save_root=opt.resized_CT_root)
    projections(data_root=opt.resized_CT_root,save_root=opt.topkMIP_CT_root)
    cat_projections_batch(data_root=opt.topkMIP_CT_root,
                          save_root=opt.batch_topkMIP_CT_root,
                          rotMat_save_root=opt.rotMat_root)

    print('Inference...')
    ####################################################################################################   
    seed_everything(opt.seed)


    ckpt = config.model.resume

    opt.resume_from_checkpoint = ckpt

    try:
        samplesdir = os.path.join(config.model.outdir, config.model.inference_name, 'samples')
        cfgdir = os.path.join(config.model.outdir, config.model.inference_name, 'configs')
        logdir = os.path.join(config.model.outdir, config.model.inference_name, 'loggertf')

        # if os.path.exists(samplesdir):
        #     continue

        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        trainer_config["gpus"] = torch.cuda.device_count()
        # default to ddp
        trainer_config["accelerator"] = "cuda"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        if not opt.train:
            config.model.params.ckpt_path = opt.resume_from_checkpoint
        model = instantiate_from_config(config.model)
        model = model.cuda()

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "tensorboard": {
                "target": "lightning.pytorch.loggers.TensorBoardLogger",
                "params": {
                    "save_dir": logdir,
                    # "name": os.path.join(logdir, 'loggertf')
                }
            }
        }
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "inference.SetupCallback",
                "params": {
                    "resume": config.model.resume,
                    "name": config.model.inference_name,
                    "logdir": logdir,
                    "ckptdir": None,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "cuda_callback": {
                "target": "inference.CUDACallback"
            },
        }
        # if version.parse(pl.__version__) >= version.parse('1.4.0'):
        #     default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()


        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        # data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = 1#len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        model.samplesdir = samplesdir
        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        # import signal

        # signal.signal(signal.SIGUSR1, melk)
        # signal.signal(signal.SIGUSR2, divein)

        # run
        # if opt.train:
        #     try:
        #         trainer.fit(model, data)
        #     except Exception:
        #         melk()
        #         raise
        # if not opt.no_test and not trainer.interrupted:
        if not opt.train:
            trainer.test(model, data)
        ## Complete the inference using the n_idx model in a single fold
        # break 
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())


    print('System matrix generation...')
    #################################################################################################### 
    from system_matrix import system_matrix_creation

    sysmat_num_splits = 30
    if len(os.listdir(opt.systemMatrix_root))==30*2:
        print('System matrix has already existed!!!')
    else:
        system_matrix_creation(save_root=opt.systemMatrix_root, sysmat_num_splits=sysmat_num_splits)

    
    print('FBP Reconstruction and optimization...')
    #################################################################################################### 
    from reconstruction import recon

    case_name_list = [item.split('.')[0] for item in os.listdir(opt.CT_root)]
    case_name_list = sorted(case_name_list)
    recon(key_in_config=opt.base, 
          data_root=os.path.join(config.model.outdir, config.model.inference_name, 'samples'), 
          case_name_list=case_name_list,
          RotMat_root=opt.rotMat_root,
          proj_save_root=os.path.join(config.model.outdir, config.model.inference_name, 'projections'),
          recon_save_root=os.path.join(config.model.outdir, config.model.inference_name, 'recons'),
          reconOpt_save_root=os.path.join(config.model.outdir, config.model.inference_name, 'reconsOpt'),
          reconOptBinary_save_root=os.path.join(config.model.outdir, config.model.inference_name, f'reconsOptBinaryPercent{opt.percentile}'),
          resized_CT_root=opt.resized_CT_root,
          sysmat_root=opt.systemMatrix_root,
          percent=opt.percentile)

    print('Noise Cancel...')
    #################################################################################################### 
    from noiseCancel import noiseCancel

    noiseCancel(data_root=os.path.join(config.model.outdir, config.model.inference_name, f'reconsOptBinaryPercent{opt.percentile}'),
                save_root=os.path.join(config.model.outdir, config.model.inference_name, 'reconsOptNoiseCancel'),
                iter_idx=4)