"""autogenerated by genpy from gps_agent_pkg/ControllerParams.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import std_msgs.msg
import gps_agent_pkg.msg

class ControllerParams(genpy.Message):
  _md5sum = "c4fa8bfed80d3d3de7dae4ff5d53bbcc"
  _type = "gps_agent_pkg/ControllerParams"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """int8 CAFFE_CONTROLLER = 1
int8 LIN_GAUSS_CONTROLLER = 2
int8 controller_to_execute

CaffeParams caffe
LinGaussParams lingauss
================================================================================
MSG: gps_agent_pkg/CaffeParams
string caffemodel # Serialized Caffe neural network
string model_prototxt # Filename of network definition file
================================================================================
MSG: gps_agent_pkg/LinGaussParams
# Time-varying Linear Gaussian controller
# Keep T copies of each
std_msgs/Float64MultiArray K_t  # Should be T x Du x Dx
std_msgs/Float64MultiArray k_t  # Should by T x Du

================================================================================
MSG: std_msgs/Float64MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float64[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding bytes at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = "height"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = "width"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = "channel"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.
================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
uint32 stride  # stride of given dimension
"""
  # Pseudo-constants
  CAFFE_CONTROLLER = 1
  LIN_GAUSS_CONTROLLER = 2

  __slots__ = ['controller_to_execute','caffe','lingauss']
  _slot_types = ['int8','gps_agent_pkg/CaffeParams','gps_agent_pkg/LinGaussParams']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       controller_to_execute,caffe,lingauss

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(ControllerParams, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.controller_to_execute is None:
        self.controller_to_execute = 0
      if self.caffe is None:
        self.caffe = gps_agent_pkg.msg.CaffeParams()
      if self.lingauss is None:
        self.lingauss = gps_agent_pkg.msg.LinGaussParams()
    else:
      self.controller_to_execute = 0
      self.caffe = gps_agent_pkg.msg.CaffeParams()
      self.lingauss = gps_agent_pkg.msg.LinGaussParams()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      buff.write(_struct_b.pack(self.controller_to_execute))
      _x = self.caffe.caffemodel
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.caffe.model_prototxt
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.lingauss.K_t.layout.dim)
      buff.write(_struct_I.pack(length))
      for val1 in self.lingauss.K_t.layout.dim:
        _x = val1.label
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_struct_2I.pack(_x.size, _x.stride))
      buff.write(_struct_I.pack(self.lingauss.K_t.layout.data_offset))
      length = len(self.lingauss.K_t.data)
      buff.write(_struct_I.pack(length))
      pattern = '<%sd'%length
      buff.write(struct.pack(pattern, *self.lingauss.K_t.data))
      length = len(self.lingauss.k_t.layout.dim)
      buff.write(_struct_I.pack(length))
      for val1 in self.lingauss.k_t.layout.dim:
        _x = val1.label
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_struct_2I.pack(_x.size, _x.stride))
      buff.write(_struct_I.pack(self.lingauss.k_t.layout.data_offset))
      length = len(self.lingauss.k_t.data)
      buff.write(_struct_I.pack(length))
      pattern = '<%sd'%length
      buff.write(struct.pack(pattern, *self.lingauss.k_t.data))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.caffe is None:
        self.caffe = gps_agent_pkg.msg.CaffeParams()
      if self.lingauss is None:
        self.lingauss = gps_agent_pkg.msg.LinGaussParams()
      end = 0
      start = end
      end += 1
      (self.controller_to_execute,) = _struct_b.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.caffe.caffemodel = str[start:end].decode('utf-8')
      else:
        self.caffe.caffemodel = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.caffe.model_prototxt = str[start:end].decode('utf-8')
      else:
        self.caffe.model_prototxt = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.lingauss.K_t.layout.dim = []
      for i in range(0, length):
        val1 = std_msgs.msg.MultiArrayDimension()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.label = str[start:end].decode('utf-8')
        else:
          val1.label = str[start:end]
        _x = val1
        start = end
        end += 8
        (_x.size, _x.stride,) = _struct_2I.unpack(str[start:end])
        self.lingauss.K_t.layout.dim.append(val1)
      start = end
      end += 4
      (self.lingauss.K_t.layout.data_offset,) = _struct_I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sd'%length
      start = end
      end += struct.calcsize(pattern)
      self.lingauss.K_t.data = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.lingauss.k_t.layout.dim = []
      for i in range(0, length):
        val1 = std_msgs.msg.MultiArrayDimension()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.label = str[start:end].decode('utf-8')
        else:
          val1.label = str[start:end]
        _x = val1
        start = end
        end += 8
        (_x.size, _x.stride,) = _struct_2I.unpack(str[start:end])
        self.lingauss.k_t.layout.dim.append(val1)
      start = end
      end += 4
      (self.lingauss.k_t.layout.data_offset,) = _struct_I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sd'%length
      start = end
      end += struct.calcsize(pattern)
      self.lingauss.k_t.data = struct.unpack(pattern, str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      buff.write(_struct_b.pack(self.controller_to_execute))
      _x = self.caffe.caffemodel
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.caffe.model_prototxt
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.lingauss.K_t.layout.dim)
      buff.write(_struct_I.pack(length))
      for val1 in self.lingauss.K_t.layout.dim:
        _x = val1.label
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_struct_2I.pack(_x.size, _x.stride))
      buff.write(_struct_I.pack(self.lingauss.K_t.layout.data_offset))
      length = len(self.lingauss.K_t.data)
      buff.write(_struct_I.pack(length))
      pattern = '<%sd'%length
      buff.write(self.lingauss.K_t.data.tostring())
      length = len(self.lingauss.k_t.layout.dim)
      buff.write(_struct_I.pack(length))
      for val1 in self.lingauss.k_t.layout.dim:
        _x = val1.label
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_struct_2I.pack(_x.size, _x.stride))
      buff.write(_struct_I.pack(self.lingauss.k_t.layout.data_offset))
      length = len(self.lingauss.k_t.data)
      buff.write(_struct_I.pack(length))
      pattern = '<%sd'%length
      buff.write(self.lingauss.k_t.data.tostring())
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.caffe is None:
        self.caffe = gps_agent_pkg.msg.CaffeParams()
      if self.lingauss is None:
        self.lingauss = gps_agent_pkg.msg.LinGaussParams()
      end = 0
      start = end
      end += 1
      (self.controller_to_execute,) = _struct_b.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.caffe.caffemodel = str[start:end].decode('utf-8')
      else:
        self.caffe.caffemodel = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.caffe.model_prototxt = str[start:end].decode('utf-8')
      else:
        self.caffe.model_prototxt = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.lingauss.K_t.layout.dim = []
      for i in range(0, length):
        val1 = std_msgs.msg.MultiArrayDimension()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.label = str[start:end].decode('utf-8')
        else:
          val1.label = str[start:end]
        _x = val1
        start = end
        end += 8
        (_x.size, _x.stride,) = _struct_2I.unpack(str[start:end])
        self.lingauss.K_t.layout.dim.append(val1)
      start = end
      end += 4
      (self.lingauss.K_t.layout.data_offset,) = _struct_I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sd'%length
      start = end
      end += struct.calcsize(pattern)
      self.lingauss.K_t.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.lingauss.k_t.layout.dim = []
      for i in range(0, length):
        val1 = std_msgs.msg.MultiArrayDimension()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.label = str[start:end].decode('utf-8')
        else:
          val1.label = str[start:end]
        _x = val1
        start = end
        end += 8
        (_x.size, _x.stride,) = _struct_2I.unpack(str[start:end])
        self.lingauss.k_t.layout.dim.append(val1)
      start = end
      end += 4
      (self.lingauss.k_t.layout.data_offset,) = _struct_I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sd'%length
      start = end
      end += struct.calcsize(pattern)
      self.lingauss.k_t.data = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
_struct_b = struct.Struct("<b")
_struct_2I = struct.Struct("<2I")