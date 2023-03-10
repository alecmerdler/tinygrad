import os, time, ctypes, hashlib, subprocess, platform
import numpy as np
from collections import defaultdict
from typing import Final, Dict
from tinygrad.helpers import dtypes, DType
from tinygrad.ops import CompiledBuffer, RawBufferCopyIn
from tinygrad.codegen.gpu import GPUCodegen, GPULanguage

class RawMallocBuffer(RawBufferCopyIn):
  def __init__(self, size, dtype : DType):
    super().__init__(size, dtype)
    self._buf = ({dtypes.float32: ctypes.c_float, dtypes.float16: ctypes.c_int16}[dtype] * size)()
  def _buffer(self): return self._buf
  def copyin(self, x:np.ndarray): ctypes.memmove(self._buf, x.ctypes.data, x.size*np.dtype(x.dtype).itemsize)
  def toCPU(self): return np.frombuffer(self._buf, dtype=self.dtype.np)

class ClangProgram:
  kernel_cnt : Final[Dict[str, int]] = defaultdict(int)
  def __init__(self, name:str, prg:str):
    prg = "#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n" + prg
    # TODO: is there a way to not write this to disk?
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"
    if not os.path.exists(fn):
      subprocess.check_output(['clang', '-shared', '-O2', '-Wall','-Werror', '-lm', '-fPIC', '-x', 'c', '-', '-o', fn+".tmp"], input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]
  def __call__(self, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args[2:]])
    if wait: return time.monotonic()-st

class ClangCodegen(GPUCodegen):
  lang = GPULanguage(buffer_suffix=" restrict")

class ClangBuffer(CompiledBuffer):
  raw_buffer_type, codegen_type, runtime_type = RawMallocBuffer, ClangCodegen, ClangProgram
