import math
import torch
import numpy as np
import cv2
import sys

__all__ = ("rgb2lum", "tanh_range", "lerp")

'''
output states:
    0: has rewards?
    1: stopped?
    2: num steps
    3:
'''
STATE_REWARD_DIM = 0
STATE_STOPPED_DIM = 1
STATE_STEP_DIM = 2
STATE_DROPOUT_BEGIN = 3


def get_expert_file_path(expert):
    expert_path = 'data/artists/fk_%s/' % expert
    return expert_path


# From github.com/OlavHN/fast-neural-style
def instance_norm(x):
    epsilon = 1e-9
    mean = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    return (x - mean) / torch.sqrt(var + epsilon)


def enrich_image_input(cfg, net, states):
    if cfg.img_include_states:
        print(("states for enriching", states.shape))
        states = states[:, None, None, :] + (net[:, :, :, 0:1] * 0)
        net = torch.cat([net, states], axis=3)
    return net


# based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class Dict(dict):
    """
      Example:
      m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
      """

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]


def make_image_grid(images, per_row=8, padding=2):
    npad = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant', constant_values=1.0)
    assert images.shape[0] % per_row == 0
    num_rows = images.shape[0] // per_row
    image_rows = []
    for i in range(num_rows):
        image_rows.append(np.hstack(images[i * per_row:(i + 1) * per_row]))
    return np.vstack(image_rows)


def get_image_center(image):
    if image.shape[0] > image.shape[1]:
        start = (image.shape[0] - image.shape[1]) // 2
        image = image[start:start + image.shape[1], :]

    if image.shape[1] > image.shape[0]:
        start = (image.shape[1] - image.shape[0]) // 2
        image = image[:, start:start + image.shape[0]]
    return image


def rotate_image(image, angle):
    """
      Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
      (in degrees). The returned image will be large enough to hold the entire
      new image, with a black background
      """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) // 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                           [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return result


def largest_rotated_rect(w, h, angle):
    """
      Given a rectangle of size wxh that has been rotated by 'angle' (in
      radians), computes the width and height of the largest possible
      axis-aligned rectangle within the rotated rectangle.

      Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

      Converted to Python by Aaron Snoswell
      """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
    """
      Given a NumPy / OpenCV 2 image, crops it to the given width and height,
      around it's centre point
      """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


# angle: degrees
def rotate_and_crop(image, angle):
    image_width, image_height = image.shape[:2]
    image_rotated = rotate_image(image, angle)
    image_rotated_cropped = crop_around_center(image_rotated,
                                               *largest_rotated_rect(
                                                   image_width, image_height,
                                                   math.radians(angle)))
    return image_rotated_cropped


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)


def double_lrelu(x, leak=0.1):
    return torch.min(torch.max(leak * x, x), leak * x - (leak - 1))


def leaky_clamp(x, lower, upper, leak=0.1):
    x_normalized = (x - lower) / (upper - lower)
    result = torch.min(torch.max(leak * x_normalized, x_normalized), leak * x_normalized - (leak - 1))
    return result * (upper - lower) + lower


class Tee(object):

    def __init__(self, name):
        self.file = open(name, 'w+')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def write_to_file(self, data):
        self.file.write(data)

    def flush(self):
        self.file.flush()


def rgb2lum(image):
    image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                              1] + 0.06 * image[:, :, :, 2]
    return image[:, :, :, None]


def tanh01(x):
    return torch.tanh(x) * 0.5 + 0.5


# def tanh_range(l, r, initial=None):
#     def get_activation(left, right, initial):
#
#         def activation(x):
#             if initial is not None:
#                 bias = math.atanh(2 * (initial - left) / (right - left) - 1)
#             else:
#                 bias = 0
#             return tanh01(x + bias) * (right - left) + left
#
#         return activation
#
#     return get_activation(l, r, initial)


def tanh_range(l, r, initial=None):
  """确保返回的是PyTorch操作"""

  def fn(x):
    # 确保输入是PyTorch tensor
    if not isinstance(x, torch.Tensor):
      x = torch.tensor(x)
    return torch.tanh(x) * (r - l) / 2.0 + (r + l) / 2.0

  return fn

def merge_dict(a, b):
    ret = a.copy()
    for key, val in list(b.items()):
        if key in ret:
            assert False, 'Item ' + key + 'already exists'
        else:
            ret[key] = val
    return ret


def lerp(a, b, l):
    return (1 - l) * a + l * b


def read_tiff16(fn):
    import tifffile
    import numpy as np
    img = tifffile.imread(fn)
    if img.dtype == np.uint8:
        depth = 8
    elif img.dtype == np.uint16:
        depth = 16
    else:
        print("Warning: unsupported data type {}. Assuming 16-bit.", img.dtype)
        depth = 16
    return (img * (1.0 / (2 ** depth - 1))).astype(np.float32)


def load_config(config_name):
    scope = {}
    exec('from config_%s import cfg' % config_name, scope)
    return scope['cfg']
