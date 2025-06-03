# 查看当前导入模块的版本号
__version__ = '8.0.142'

from ultralytics.engine.model import YOLO
from ultralytics.hub import start
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'checks', 'download', 'start', 'settings'  # allow simpler import
