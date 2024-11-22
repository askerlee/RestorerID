from pytorch_lightning.trainer import Trainer
import torch
import argparse,os,glob,datetime
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config, instantiate_from_config_sr
from packaging import version
import pytorch_lightning as pl
import main
from RealESRGAN_CBDNet import Degradation_model
from pytorch_lightning.callbacks import ModelCheckpoint


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
    parser.add_argument("--name",type=str,const=True, default="",nargs="?",help="postfix for logdir",)
    parser.add_argument("--resume",type=str,const=True,default="",nargs="?",help="resume from logdir or checkpoint in logdir",)
    parser.add_argument("--base",nargs="*",metavar="configs/v15/v15-BaseModel.yaml",help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",default=list(),)
    parser.add_argument("--train",type=str2bool,const=True,default=False,nargs="?",help="train",)
    parser.add_argument("--no-test",type=str2bool,const=True,default=False,nargs="?",help="disable test",)
    parser.add_argument("--project",help="name of new or path to existing project")
    parser.add_argument("--debug",type=str2bool,nargs="?",const=True,default=False,help="enable post-mortem debugging",)
    parser.add_argument("--seed",type=int,default=23,help="seed for seed_everything",)
    parser.add_argument("--postfix",type=str,default="",help="post-postfix for default name",)
    parser.add_argument("--logdir",type=str,default="./logs",help="directory for logging dat shit",)
    parser.add_argument("--scale_lr",type=str2bool,nargs="?",const=True,default=False,help="scale base-lr by ngpu * batch_size * n_accumulate",)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        if os.path.isfile(opt.resume):
            ckpt = opt.resume
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
        else:
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(opt.resume, "checkpoints", "last.ckpt")
    
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        nowname = logdir.split("/")[-1]
    else:
        
        nowname = now + opt.name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    config = OmegaConf.merge(*configs)                                       
    lightning_config = config.pop("lightning", OmegaConf.create())          
    trainer_config = lightning_config.get("trainer", OmegaConf.create())    

    model = instantiate_from_config(config.model)
    model.configs = config
    model.Degrad = Degradation_model.Degradation(config.degradation)


    setup_callback = main.SetupCallback(resume=opt.resume, now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config=config, lightning_config=lightning_config)
    image_logger = main.ImageLogger(batch_frequency= 4000,max_images= 2,clamp= True, increase_log_steps= False)
    learning_rate_logger = main.LearningRateMonitor(logging_interval= "step")
    cuda_callback = main.CUDACallback()
    checkpoint_callback = ModelCheckpoint(dirpath= ckptdir,                  
                                        filename= 'model-{epoch}-{step:06d}', 
                                        verbose= True,                     
                                        save_last= False,
                                        every_n_train_steps = 20000,        
                                        )
    
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir = logdir, name = 'tensorboard')

    trainer = Trainer(gpus = trainer_config['gpus'],
                      max_steps = trainer_config['max_steps'],
                      precision = trainer_config['precision'],
                      strategy = trainer_config['strategy'],
                      benchmark = trainer_config['benchmark'],
                      accumulate_grad_batches = trainer_config['accumulate_grad_batches'],
                      default_root_dir = logdir,
                    #   resume_from_checkpoint = ckpt,
                      logger = logger,
                      callbacks= [setup_callback,image_logger,learning_rate_logger,cuda_callback,checkpoint_callback],
                    )
    trainer.logdir = logdir

    dataloader = instantiate_from_config(config.data)
    dataloader.prepare_data()
    dataloader.setup()

    for k in dataloader.datasets:
        print(f"{k}, {dataloader.datasets[k].__class__.__name__}, {len(dataloader.datasets[k])}")
    
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches

    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    else:
        model.learning_rate = base_lr
    
    trainer.fit(model, dataloader)


    






