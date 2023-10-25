# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
import sys
from pathlib import Path
from typing import Union

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.hub.utils import HUB_WEB_ROOT
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, callbacks, checks, emojis, yaml_load
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Model(nn.Module):
    """
    A base class to unify APIs for all models.
    #// 用于统一所有模型的API的基类
    Args:
        model (str, Path): Path to the model file to load or create.
        task (Any, optional): Task type for the YOLO model. Defaults to None.

    Attributes:
        predictor (Any): The predictor object.
        model (Any): The model object.
        trainer (Any): The trainer object.
        task (str): The type of model task.
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.
        cfg (str): The model configuration if loaded from *.yaml file.
        ckpt_path (str): The checkpoint file path.
        overrides (dict): Overrides for the trainer object.
        metrics (Any): The data for metrics.

    Methods:
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method.
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
        _load(weights:str, task:str='') -> None:
            Initializes a new model and infers the task type from the model head.
        _check_is_pytorch_model() -> None:
            Raises TypeError if the model is not a PyTorch model.
        reset() -> None:
            Resets the model modules.
        info(verbose:bool=False) -> None:
            Logs the model info.
        fuse() -> None:
            Fuses the model for faster inference.
        predict(source=None, stream=False, **kwargs) -> List[ultralytics.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(ultralytics.engine.results.Results): The prediction results.
    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt                 #// 权重文件
        self.cfg = None  # if loaded from *.yaml                #// 配置文件
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object     #// 用于训练参数重新设置
        self.metrics = None  # validation/training metrics      #// 评价指标
        self.session = None  # HUB session                      #// HUB会话
        self.task = task  # task type                           #// 模型任务

        #// model表示路径，可以是权重路径也可以是model配置文件.yaml文件路径
        model = str(model).strip()  # strip spaces

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):                #// 检查模型是否是hub model
            from ultralytics.hub.session import HUBTrainingSession
            self.session = HUBTrainingSession(model)
            model = self.session.model_file

        # Check if Triton Server model
        elif self.is_triton_model(model):           #// 检查模型是否是triton server model
            self.model = model
            self.task = task
            return                                  #// 如果是triton模型，直接返回

        # Load or create new YOLO model
        suffix = Path(model).suffix                 #// 获取model的后缀
        #// 如果没有后缀并且model的名字在预先配置中，则为model添加pt后缀，并配置suffix
        if not suffix and Path(model).stem in GITHUB_ASSETS_STEMS:
            model, suffix = Path(model).with_suffix('.pt'), '.pt'  # add suffix, i.e. yolov8n -> yolov8n.pt

        #// 如果suffix为yaml或yml调用_new()函数构建model
        if suffix in ('.yaml', '.yml'):
            self._new(model, task)
        #// 否则直接通过_load()函数加载权重
        else:
            self._load(model, task)
    
    #// __call__()函数将类的实例化对象转化为可调用对象
    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        #// 通过对Model类对象传入相关参数，执行predict()函数
        return self.predict(source, stream, **kwargs)

    #// 判断模型是否为triton模型
    @staticmethod
    def is_triton_model(model):
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>"""
        from urllib.parse import urlsplit
        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {'http', 'grfc'}

    #// 判断模型是否为hub模型
    @staticmethod
    def is_hub_model(model):
        """Check if the provided model is a HUB model."""
        return any((
            model.startswith(f'{HUB_WEB_ROOT}/models/'),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
            [len(x) for x in model.split('_')] == [42, 20],  # APIKEY_MODELID
            len(model) == 20 and not Path(model).exists() and all(x not in model for x in './\\')))  # MODELID

    #// 从模型配置yaml文件中初始化模型
    def _new(self, cfg: str, task=None, model=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.
        Args:
            cfg (str): model configuration file         #// 模型配置文件
            task (str | None): model task               #// 执行任务名称
            model (BaseModel): Customized model.        #// 任务相对应的模型
            verbose (bool): display model info on load  #// 是否显示模型信息
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)  #// 如果指定task直接使用，否则通过model配置文件中head推断task
        #// 给定model直接使用，否则通过task智能加载模型
        self.model = (model or self._smart_load('model'))(cfg_dict, verbose=verbose and RANK == -1)  # build model

        self.overrides['model'] = self.cfg              #// 将overrides中“model”设置为模型配置文件路径
        self.overrides['task'] = self.task              #// 任务名称

        # Below added to allow export from YAMLs
        #// 更新模型配置
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        #// 设置模型执行的任务
        self.model.task = self.task

    #// 从权重文件初始化模型
    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.
        #// 初始化新模型并从模型检测头推断任务类型
        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix                               #// 获取模型的后缀
        #// 如果模型后缀为.pt
        if suffix == '.pt':
            #// ckpk：保存文件中的模型以及训练配置信息，此处ckpt是包含model信息的
            self.model, self.ckpt = attempt_load_one_weight(weights)                   #// 返回模型以及权重文件名称
            self.task = self.model.args['task']                                        #// 获取模型的task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            #todo
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        
        self.overrides['model'] = weights                   #// 将overrides中"model"设置为权重路径
        self.overrides['task'] = self.task                  #// 设置task

    #// 判断模型是否为pytorch模型
    def _check_is_pytorch_model(self):
        """Raises TypeError is model is not a PyTorch model."""
        #// 是否为权重路径，并且后缀为.pt
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'
        #// 是否已经是构建好的模型
        pt_module = isinstance(self.model, nn.Module)       
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'")

    #// 将模型中指定参数重新设置为随机初始化值，抛弃该模块的训练信息
    def reset_weights(self):
        """Resets the model modules parameters to randomly initialized values, losing all training information."""
        self._check_is_pytorch_model()              #// 检查是否为pytorch model
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):      #// 如果模块有reset_parameters参数，则抛弃该模块的训练信息
                m.reset_parameters()
        for p in self.model.parameters():           #// 将模块设置为梯度可更新
            p.requires_grad = True
        return self

    #// 函数没有调用过
    # def load(self, weights='yolov8n.pt'):
    #     """Transfers parameters with matching names and shapes from 'weights' to model.
    #     #// 将具有匹配名称和形状的参数,从给定权重赋值给model
    #     """
    #     self._check_is_pytorch_model()
    #     if isinstance(weights, (str, Path)):
    #         weights, self.ckpt = attempt_load_one_weight(weights)
    #     self.model.load(weights)
    #     return self

    def info(self, detailed=False, verbose=True):
        """
        #// 记录模型的详细信息
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    #// 执行模型算子融合
    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self._check_is_pytorch_model()
        self.model.fuse()

    #// 模型预测模块
    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        """
        Perform prediction using the YOLO model.
        #// 使用YOLO模型进行预测
        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            predictor (BasePredictor): Customized predictor.
            **kwargs : Additional keyword arguments passed to the predictor.
                Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        #// 如果待预测文件source为空，则将默认文件夹作为预测source文件
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        #// 判断是否是命令行命令
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))

        #// 置信度阈值，是否保存的参数
        custom = {'conf': 0.25, 'save': is_cli}  # method defaults
        #// 配置参数，优先级较高的在最后
        args = {**self.overrides, **custom, **kwargs, 'mode': 'predict'}  # highest priority args on the right
        #// 提示词，只针对SAM模型
        prompts = args.pop('prompts', None)  # for SAM-type models

        #// 如果没有给定predictor，根据task智能加载predictor
        if not self.predictor:
            self.predictor = (predictor or self._smart_load('predictor'))(overrides=args, _callbacks=self.callbacks)
            #// setup_model设置模型参数
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            #// 如果predictor已经给定，则指更新其参数
            self.predictor.args = get_cfg(self.predictor.args, args)
            #// 获取文件保存地址
            if 'project' in args or 'name' in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        #// SAM model时使用
        if prompts and hasattr(self.predictor, 'set_prompts'):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        #todo 返回预测结果
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    #// 使用跟踪器对输入进行跟踪
    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.
        #// 使用已注册的跟踪器对输入源执行对象跟踪

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): The tracking results.
        """
        if not hasattr(self.predictor, 'trackers'):
            from ultralytics.trackers import register_tracker
            register_tracker(self, persist)
        kwargs['conf'] = kwargs.get('conf') or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    #// 验证模型指标
    def val(self, validator=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            validator (BaseValidator): Customized validator.
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        #// 默认验证过程，使用矩形推理
        custom = {'rect': True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'val'}  # highest priority args on the right

        #// 加载验证器验证器
        validator = (validator or self._smart_load('validator'))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    #// 导出模型并且进行基准测试（使用导出格式进行推理，验证指标）
    def benchmark(self, **kwargs):
        """
        Benchmark a model on all export formats.
        #// 对所有导出格式的模型进行基准测试
        Args:
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {'verbose': False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, 'mode': 'benchmark'}
        return benchmark(
            model=self,
            data=kwargs.get('data'),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args['imgsz'],
            half=args['half'],
            int8=args['int8'],
            device=args['device'],
            verbose=kwargs.get('verbose'))

    #// 导出模型
    def export(self, **kwargs):
        """
        Export model.
        #// 导出模型
        Args:
            **kwargs : Any other args accepted by the Exporter. To see all args check 'configuration' section in docs.
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {'imgsz': self.model.args['imgsz'], 'batch': 1, 'data': None, 'verbose': False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'export'}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    #// 执行模型训练过程
    def train(self, trainer=None, **kwargs):
        """
        Trains the model on a given dataset.
        #// 在给定的数据集训练模型
        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        #// 判断模型是否为pytorch模型，如果不是引发错误
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        checks.check_pip_update_available()             #// 检查库文件
        
        #// 如果给定参数中存在cfg参数则加载配置文件，否则使用self.overrides中的参数
        overrides = yaml_load(checks.check_yaml(kwargs['cfg'])) if kwargs.get('cfg') else self.overrides
        custom = {'data': TASK2DATA[self.task]}  # method defaults          #// 默认data配置文件
        #// 合并配置文件
        args = {**overrides, **custom, **kwargs, 'mode': 'train'}  # highest priority args on the right
        if args.get('resume'):
            args['resume'] = self.ckpt_path

        self.trainer = (trainer or self._smart_load('trainer'))(overrides=args, _callbacks=self.callbacks)
        if not args.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP
        return self.metrics

    #//todo 超参数优化
    def tune(self, use_ray=False, iterations=10, *args, **kwargs):
        """
        Runs hyperparameter tuning, optionally using Ray Tune. See ultralytics.utils.tuner.run_ray_tune for Args.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, 'mode': 'train'}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    #todo
    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides['device'] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    #// 获取模型检测类别名称
    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, 'names') else None

    #// 获取模型所在的device
    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    #// 返回模型的transform
    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, 'transforms') else None

#todo ---------------------------------回调函数部分------------------------------------------
    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        """Clear all event callbacks."""
        self.callbacks[event] = []

    def reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]
#todo ----------------------------------------------------------------------------------------

    #// 加载pytorch模型时，从args参数中获取部分参数
    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    #// 根据task参数，以及任务组件名称，智能加载对应的组件
    def _smart_load(self, key):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")) from e

    #// 任务集合，包含执行任务需要组件：model，trainer，validator，predictor
    @property
    def task_map(self):
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError('Please provide task map for your model!')
