# Ultralytics YOLO 🚀, AGPL-3.0 license
import sys
from pathlib import Path
from typing import Union
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel, SegmentationModel,
                                  attempt_load_one_weight, guess_model_task, nn, yaml_model_load)
from ultralytics.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, callbacks,
                               is_git_dir, yaml_load)
from ultralytics.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.utils.downloads import GITHUB_ASSET_STEMS
from ultralytics.utils.torch_utils import smart_inference_mode


# Map head to model, trainer, validator, and predictor classes
TASK_MAP = {
    'classify': [
        ClassificationModel, yolo.classify.ClassificationTrainer, yolo.classify.ClassificationValidator,
        yolo.classify.ClassificationPredictor],
    'detect':
    [DetectionModel, yolo.detect.DetectionTrainer, yolo.detect.DetectionValidator, yolo.detect.DetectionPredictor],
    'segment': [
        SegmentationModel, yolo.segment.SegmentationTrainer, yolo.segment.SegmentationValidator,
        yolo.segment.SegmentationPredictor],
    'pose': [PoseModel, yolo.pose.PoseTrainer, yolo.pose.PoseValidator, yolo.pose.PosePredictor]}


class YOLO:
    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        # 拿到默认的回调函数的字典
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        model = str(model).strip()  # strip spaces

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        # is_hub_model是false
        if self.is_hub_model(model):
            from ultralytics.hub.session import HUBTrainingSession
            self.session = HUBTrainingSession(model)
            model = self.session.model_file

        # Load or create new YOLO model
        # 拿到的是yaml后缀
        suffix = Path(model).suffix
        # 判断suffix不再stem之中而主干部分在
        # add suffix, i.e. yolov8n -> yolov8n.pt
        # 处理不加suffix的情况
        if not suffix and Path(model).stem in GITHUB_ASSET_STEMS:
            model, suffix = Path(model).with_suffix('.pt'), '.pt'
        # 如果是.yaml的文件格式则使用scratsh从头训练
        if suffix == '.yaml':
            self._new(model, task)
        else:
            self._load(model, task)

    # call mothed is first
    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    @staticmethod
    def is_hub_model(model):
        """Check if the provided model is a HUB model."""
        return any((
            model.startswith('https://hub.ultralytics.com/models/'),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
            [len(x) for x in model.split('_')] == [42, 20],  # APIKEY_MODELID
            len(model) == 20 and not Path(model).exists() and all(x not in model for x in './\\')))  # MODELID

    def _new(self, cfg: str, task=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            verbose (bool): display model info on load
        """
        # 返回字典d
        # # {'nc': 6, 'scales': [1.0, 1.0, 512], 'backbone':[], "head":[], "yaml_file": "./././"}
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        # detect
        self.task = task or guess_model_task(cfg_dict)
        # DetectionModel(cfg_dict, verbose=verbose and RANK == -1)
        # 这里可以添加一个nc参数
        self.model = TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)  # build model
        # 再次使用字典封装一个模型框架
        self.overrides['model'] = self.cfg
        # Below added to allow export from yamls
        # 将模型的骨架文件和模型的超参数文件（defult.yaml）结合，形成一个全体参数的字典
        # combine model and default args, preferring model args
        args = {**DEFAULT_CFG_DICT, **self.overrides}
        # 把defualt.yaml中所有非None的键值对拿到
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        # 定义detect任务
        self.model.task = self.task

    # 预训练的模型加载
    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """

        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    def _check_is_pytorch_model(self):
        """
        Raises TypeError is model is not a PyTorch model
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(f"model='{self.model}' must be a *.pt PyTorch model, but is a different type. "
                            f'PyTorch models can be used to train, val, predict and export, i.e. '
                            f"'yolo export model=yolov8n.pt', but exported formats like ONNX, TensorRT etc. only "
                            f"support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'.")

    @smart_inference_mode()
    def reset_weights(self):
        """
        Resets the model modules parameters to randomly initialized values, losing all training information.
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    @smart_inference_mode()
    def load(self, weights='yolov8n.pt'):
        """
        Transfers parameters with matching names and shapes from 'weights' to model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self._check_is_pytorch_model()
        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        # 没有数据就去网站下载图片来做测试
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        # 用命令行来测试
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        # 置信度
        overrides['conf'] = 0.4
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = TASK_MAP[self.task][3](overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
            if 'project' in overrides or 'name' in overrides:
                self.predictor.save_dir = self.predictor.get_save_dir()
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

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
        # ByteTrack-based method needs low confidence predictions as input
        conf = kwargs.get('conf') or 0.1
        kwargs['conf'] = conf
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        # 使用自定义的参数来覆盖的配置
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TASK_MAP[self.task][2](args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics

        return validator.metrics

    @smart_inference_mode()
    def benchmark(self, **kwargs):
        """
        Benchmark a model on all export formats.

        Args:
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark
        overrides = self.model.args.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'benchmark'
        overrides = {**DEFAULT_CFG_DICT, **overrides}  # fill in missing overrides keys with defaults
        return benchmark(
            model=self,
            data=kwargs.get('data'),  # if no 'data' argument passed set data=None for default datasets
            imgsz=overrides['imgsz'],
            half=overrides['half'],
            int8=overrides['int8'],
            device=overrides['device'],
            verbose=overrides['verbose'])

    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        if overrides.get('imgsz') is None:
            overrides['imgsz'] = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if 'batch' not in kwargs:
            overrides['batch'] = 1  # default to 1 if not modified
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        check_pip_update_available()
        overrides = self.overrides.copy()
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides.update(kwargs)
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task
        self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self._check_is_pytorch_model()
        self.model.to(device)

    def tune(self, *args, **kwargs):
        """
        Runs hyperparameter tuning using Ray Tune. See ultralytics.utils.tuner.run_ray_tune for Args.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            ModuleNotFoundError: If Ray Tune is not installed.
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.tuner import run_ray_tune
        return run_ray_tune(self, *args, **kwargs)

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        """Clear all event callbacks."""
        self.callbacks[event] = []

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]
