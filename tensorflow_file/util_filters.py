import math
import tensorflow as tf


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



def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                                  1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]


def tanh01(x):
  return tf.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):

  def get_activation(left, right, initial):

    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left

    return activation

  return get_activation(l, r, initial)


def lerp(a, b, l):
  return (1 - l) * a + l * b
