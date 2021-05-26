# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Lightweight TVM RPC module.

RPC enables connect to a remote server, upload and launch functions.
This is useful to for cross-compile and remote testing,
The compiler stack runs on local server, while we use RPC server
to run on remote runtime which don't have a compiler available.

The test program compiles the program on local server,
upload and run remote RPC server, get the result back to verify correctness.
"""
from __future__ import absolute_import as _abs

from .server import Server
from .client import connect, connect_tracker
from .client import RPCSession, LocalSession, PopenSession, TrackerSession
from .minrpc import with_minrpc
import os
import logging
from tvm.contrib import cc
import ctypes
import json
import tvm
import sys
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""VTA related intrinsics"""

import tvm
from tvm import te


def gemm(env, mock=False):
    """Matrix-matrix multiply intrinsic

    Parameters
    ----------
    env : Environment
        The Environment

    mock : bool
        Whether create a mock version.
    """
    wgt_lanes = env.WGT_ELEM_BITS // env.WGT_WIDTH
    assert wgt_lanes == env.BLOCK_OUT * env.BLOCK_IN
    wgt_shape = (env.BLOCK_OUT, env.BLOCK_IN)
    assert wgt_shape[0] * wgt_shape[1] == wgt_lanes

    inp_lanes = env.INP_ELEM_BITS // env.INP_WIDTH
    assert inp_lanes == env.BATCH * env.BLOCK_IN
    inp_shape = (env.BATCH, env.BLOCK_IN)
    assert inp_shape[0] * inp_shape[1] == inp_lanes

    out_lanes = env.ACC_ELEM_BITS // env.ACC_WIDTH
    assert out_lanes == env.BATCH * env.BLOCK_OUT
    out_shape = (env.BATCH, env.BLOCK_OUT)
    assert out_shape[0] * out_shape[1] == out_lanes

    wgt = te.placeholder(
        (wgt_shape[0], wgt_shape[1]), dtype="int%d" % env.WGT_WIDTH, name=env.wgt_scope
    )
    inp = te.placeholder(
        (inp_shape[0], inp_shape[1]), dtype="int%d" % env.INP_WIDTH, name=env.inp_scope
    )
    k = te.reduce_axis((0, wgt_shape[1]), name="k")
    out_dtype = "int%d" % env.ACC_WIDTH
    out = te.compute(
        (out_shape[0], out_shape[1]),
        lambda i, j: te.sum(inp[i, k].astype(out_dtype) * wgt[j, k].astype(out_dtype), axis=[k]),
        name="out",
    )
    wgt_layout = tvm.tir.decl_buffer(
        wgt.shape,
        wgt.dtype,
        env.wgt_scope,
        scope=env.wgt_scope,
        offset_factor=wgt_lanes,
        data_alignment=wgt_lanes,
    )
    inp_layout = tvm.tir.decl_buffer(
        inp.shape,
        inp.dtype,
        env.inp_scope,
        scope=env.inp_scope,
        offset_factor=inp_lanes,
        data_alignment=inp_lanes,
    )
    out_layout = tvm.tir.decl_buffer(
        out.shape,
        out.dtype,
        env.acc_scope,
        scope=env.acc_scope,
        offset_factor=out_lanes,
        data_alignment=out_lanes,
    )

    def intrin_func(ins, outs):
        """Matrix-matrix multiply intrinsic function"""
        dinp, dwgt = ins
        dout = outs[0]

        def instr(index):
            """Generate matrix-matrix multiply VTA instruction"""
            irb = tvm.tir.ir_builder.create()
            dev = env.dev
            irb.scope_attr(dev.vta_axis, "coproc_scope", dev.get_task_qid(dev.QID_COMPUTE))
            irb.scope_attr(dev.vta_axis, "coproc_uop_scope", dev.vta_push_uop)
            if index in (0, 2):
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32",
                        "tir.vta.uop_push",
                        0,
                        0,
                        dout.access_ptr("rw", "int32"),
                        dinp.access_ptr("r", "int32"),
                        dwgt.access_ptr("r", "int32"),
                        0,
                        0,
                        0,
                    )
                )
            else:
                irb.emit(
                    tvm.tir.call_intrin(
                        "int32",
                        "tir.vta.uop_push",
                        0,
                        1,
                        dout.access_ptr("rw", "int32"),
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                )
            return irb.get()

        # return a triple of normal-set, reset, update
        nop = tvm.tir.Evaluate(0)
        if mock:
            return (nop, nop, nop)
        return (instr(0), instr(1), instr(2))

    return te.decl_tensor_intrin(
        out.op, intrin_func, name="GEMM", binds={inp: inp_layout, wgt: wgt_layout, out: out_layout}
    )

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""VTA specific bitstream program library."""
import os
import argparse


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str, default="", help="target")
    parser.add_argument("bitstream", type=str, default="", help="bitstream path")
    args = parser.parse_args()

    if args.target not in ("pynq", "ultra96", "de10nano", "sim", "tsim"):
        raise RuntimeError("Unknown target {}".format(args.target))

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    path_list = [
        os.path.join(curr_path, "/{}".format(args.bitstream)),
        os.path.join("./", "{}".format(args.bitstream)),
    ]
    ok_path_list = [p for p in path_list if os.path.exists(p)]
    if not ok_path_list:
        raise RuntimeError("Cannot find bitstream file in %s" % str(path_list))

    bitstream_program(args.target, args.bitstream)


def pynq_bitstream_program(bitstream_path):
    # pylint: disable=import-outside-toplevel
    from pynq import Bitstream

    bitstream = Bitstream(bitstream_path)
    bitstream.download()


def de10nano_bitstream_program(bitstream_path):
    # pylint: disable=import-outside-toplevel
    from tvm import get_global_func

    program = get_global_func("vta.de10nano.program")
    program(bitstream_path)


def intelfocl_bitstream_program(bitstream_path, mem_size=4 * 1024 * 1024 * 1024):
    # pylint: disable=import-outside-toplevel
    from tvm import get_global_func

    program = get_global_func("vta.oclfpga.program")
    program(bitstream_path, mem_size)


def bitstream_program(target, bitstream, *args):
    """program bitstream to devices"""

    if target in ["pynq", "ultra96"]:
        pynq_bitstream_program(bitstream)
    elif target in ["de10nano"]:
        de10nano_bitstream_program(bitstream)
    elif target in ["sim", "tsim"]:
        # In simulation, bit stream programming is a no-op
        return
    elif target in ["intelfocl"]:
        intelfocl_bitstream_program(bitstream, *args)
    else:
        raise RuntimeError("Unknown target {}".format(target))


if __name__ == "__main__":
    main()


# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Configurable VTA Hareware Environment scope."""
# pylint: disable=invalid-name, exec-used


import os
import json
import copy
import tvm
from tvm import te
from tvm.ir.op import register_intrin_lowering
# from . import intrin


def get_vta_hw_path():
    """Get the VTA HW path."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    vta_hw_default = os.path.abspath(os.path.join(curr_path, "../../../3rdparty/vta-hw"))
    VTA_HW_PATH = os.getenv("VTA_HW_PATH", vta_hw_default)
    return os.path.abspath(VTA_HW_PATH)


def pkg_config(cfg):
    """Returns PkgConfig pkg config object."""
    pkg_config_py = os.path.join(get_vta_hw_path(), "config/pkg_config.py")
    libpkg = {"__file__": pkg_config_py}
    exec(compile(open(pkg_config_py, "rb").read(), pkg_config_py, "exec"), libpkg, libpkg)
    PkgConfig = libpkg["PkgConfig"]
    return PkgConfig(cfg)


class DevContext(object):
    """Internal development context

    This contains all the non-user facing compiler
    internal context that is hold by the Environment.

    Parameters
    ----------
    env : Environment
        The environment hosting the DevContext

    Note
    ----
    This class is introduced so we have a clear separation
    of developer related, and user facing attributes.
    """

    # Memory id for DMA
    MEM_ID_UOP = 0
    MEM_ID_WGT = 1
    MEM_ID_INP = 2
    MEM_ID_ACC = 3
    MEM_ID_OUT = 4
    MEM_ID_ACC_8BIT = 5
    # VTA ALU Opcodes
    ALU_OPCODE_MIN = 0
    ALU_OPCODE_MAX = 1
    ALU_OPCODE_ADD = 2
    ALU_OPCODE_SHR = 3
    ALU_OPCODE_MUL = 4
    # Task queue id (pipeline stage)
    QID_LOAD_INP = 1
    QID_LOAD_WGT = 1
    QID_LOAD_OUT = 2
    QID_STORE_OUT = 3
    QID_COMPUTE = 2

    def __init__(self, env):
        self.vta_axis = te.thread_axis("vta")
        self.vta_push_uop = tvm.tir.StringImm("VTAPushGEMMOp")
        ctx = tvm.tir.call_intrin("handle", "tir.vta.command_handle")
        self.command_handle = tvm.tir.Call("handle", "tir.tvm_thread_context", [ctx])
        self.DEBUG_NO_SYNC = False
        env._dev_ctx = self
        self.gemm = gemm(env, env.mock_mode)

    def get_task_qid(self, qid):
        """Get transformed queue index."""
        return 1 if self.DEBUG_NO_SYNC else qid


class Environment(object):
    """Hardware configuration object.

    This object contains all the information
    needed for compiling to a specific VTA backend.

    Parameters
    ----------
    cfg : dict of str to value.
        The configuration parameters.

    Example
    --------
    .. code-block:: python

      # the following code reconfigures the environment
      # temporarily to attributes specified in new_cfg.json
      new_cfg = json.load(json.load(open("new_cfg.json")))
      with vta.Environment(new_cfg):
          # env works on the new environment
          env = vta.get_env()
    """

    current = None
    # constants
    MAX_XFER = 1 << 22
    # debug flags
    DEBUG_DUMP_INSN = 1 << 1
    DEBUG_DUMP_UOP = 1 << 2
    DEBUG_SKIP_READ_BARRIER = 1 << 3
    DEBUG_SKIP_WRITE_BARRIER = 1 << 4
    # memory scopes
    inp_scope = "local.inp_buffer"
    wgt_scope = "local.wgt_buffer"
    acc_scope = "local.acc_buffer"

    # initialization function
    def __init__(self, cfg):
        # Produce the derived parameters and update dict
        self.pkg = pkg_config(cfg)
        self.__dict__.update(self.pkg.cfg_dict)
        # data type width
        self.INP_WIDTH = 1 << self.LOG_INP_WIDTH
        self.WGT_WIDTH = 1 << self.LOG_WGT_WIDTH
        self.ACC_WIDTH = 1 << self.LOG_ACC_WIDTH
        self.OUT_WIDTH = 1 << self.LOG_OUT_WIDTH
        # tensor intrinsic shape
        self.BATCH = 1 << self.LOG_BATCH
        self.BLOCK_IN = 1 << self.LOG_BLOCK_IN
        self.BLOCK_OUT = 1 << self.LOG_BLOCK_OUT
        # buffer size
        self.UOP_BUFF_SIZE = 1 << self.LOG_UOP_BUFF_SIZE
        self.INP_BUFF_SIZE = 1 << self.LOG_INP_BUFF_SIZE
        self.WGT_BUFF_SIZE = 1 << self.LOG_WGT_BUFF_SIZE
        self.ACC_BUFF_SIZE = 1 << self.LOG_ACC_BUFF_SIZE
        self.OUT_BUFF_SIZE = 1 << self.LOG_OUT_BUFF_SIZE
        # bytes per buffer
        self.INP_ELEM_BITS = self.BATCH * self.BLOCK_IN * self.INP_WIDTH
        self.WGT_ELEM_BITS = self.BLOCK_OUT * self.BLOCK_IN * self.WGT_WIDTH
        self.ACC_ELEM_BITS = self.BATCH * self.BLOCK_OUT * self.ACC_WIDTH
        self.OUT_ELEM_BITS = self.BATCH * self.BLOCK_OUT * self.OUT_WIDTH
        self.INP_ELEM_BYTES = self.INP_ELEM_BITS // 8
        self.WGT_ELEM_BYTES = self.WGT_ELEM_BITS // 8
        self.ACC_ELEM_BYTES = self.ACC_ELEM_BITS // 8
        self.OUT_ELEM_BYTES = self.OUT_ELEM_BITS // 8
        # dtypes
        self.acc_dtype = "int%d" % self.ACC_WIDTH
        self.inp_dtype = "int%d" % self.INP_WIDTH
        self.wgt_dtype = "int%d" % self.WGT_WIDTH
        self.out_dtype = "int%d" % self.OUT_WIDTH
        # bistream name
        self.BITSTREAM = self.pkg.bitstream
        # model string
        self.MODEL = self.TARGET + "_" + self.BITSTREAM
        # lazy cached members
        self.mock_mode = False
        self._mock_env = None
        self._dev_ctx = None
        self._last_env = None

    def __enter__(self):
        self._last_env = Environment.current
        Environment.current = self
        return self

    def __exit__(self, ptype, value, trace):
        Environment.current = self._last_env

    @property
    def cfg_dict(self):
        return self.pkg.cfg_dict

    @property
    def dev(self):
        """Developer context"""
        if self._dev_ctx is None:
            self._dev_ctx = DevContext(self)
        return self._dev_ctx

    @property
    def mock(self):
        """A mock version of the Environment

        The ALU, dma_copy and intrinsics will be
        mocked to be nop.
        """
        if self.mock_mode:
            return self
        if self._mock_env is None:
            self._mock_env = copy.copy(self)
            self._mock_env._dev_ctx = None
            self._mock_env.mock_mode = True
        return self._mock_env

    @property
    def dma_copy(self):
        """DMA copy pragma"""
        return "dma_copy" if not self.mock_mode else "skip_dma_copy"

    @property
    def alu(self):
        """ALU pragma"""
        return "alu" if not self.mock_mode else "skip_alu"

    @property
    def gemm(self):
        """GEMM intrinsic"""
        return self.dev.gemm

    @property
    def target(self):
        return tvm.target.vta(model=self.MODEL)

    @property
    def target_host(self):
        """The target host"""
        if self.TARGET in ["pynq", "de10nano"]:
            return "llvm -mtriple=armv7-none-linux-gnueabihf"
        if self.TARGET == "ultra96":
            return "llvm -mtriple=aarch64-linux-gnu"
        if self.TARGET in ["sim", "tsim", "intelfocl"]:
            return "llvm"
        raise ValueError("Unknown target %s" % self.TARGET)

    @property
    def target_vta_cpu(self):
        return tvm.target.arm_cpu(model=self.TARGET)


def get_env():
    """Get the current VTA Environment.

    Returns
    -------
    env : Environment
        The current environment.
    """
    return Environment.current



def _init_env():
    """Initialize the default global env"""
    config_path = os.path.join(get_vta_hw_path(), "config/vta_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError("Cannot find config in %s" % str(config_path))
    cfg = json.load(open(config_path))
    return Environment(cfg)


Environment.current = _init_env()


def _get_lib_name(lib_name):
    """Get lib name with extension

    Returns
    -------
    lib_name_ext : str
        Name of VTA shared library with extension

    Parameters
    ------------
    lib_name : str
        Name of VTA shared library
    """
    if sys.platform.startswith("win32"):
        return lib_name + ".dll"
    if sys.platform.startswith("darwin"):
        return lib_name + ".dylib"
    return lib_name + ".so"


def find_libvta(lib_vta, optional=False):
    """Find VTA Chisel-based library

    Returns
    -------
    lib_found : str
        Library path

    Parameters
    ------------
    lib_vta : str
        Name of VTA shared library

    optional : bool
        Enable error check
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_search = [
        os.path.join(
            curr_path,
            "..",
            "..",
            "..",
            "build",
        )
    ]
    lib_search += [os.path.join(get_vta_hw_path(), "build")]
    lib_name = _get_lib_name(lib_vta)
    lib_path = [os.path.join(x, lib_name) for x in lib_search]
    lib_found = [x for x in lib_path if os.path.exists(x)]
    if not lib_found and not optional:
        raise RuntimeError(
            "Cannot find the files.\n" + "List of candidates:\n" + str("\n".join(lib_path))
        )
    return lib_found


@tvm.register_func("tvm.rpc.server.start", override=True)
def server_start():
    """VTA RPC server extension."""
    # pylint: disable=unused-variable
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../../../"))
    dll_path = find_libvta("libvta")[0]
    cfg_path = os.path.abspath(os.path.join(
        proj_root, "3rdparty/vta-hw/config/vta_config.json"))
    runtime_dll = []
    _load_module = tvm.get_global_func("tvm.rpc.server.load_module")

    def load_vta_dll():
        """Try to load vta dll"""
        if not runtime_dll:
            runtime_dll.append(ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL))
        logging.info("Loading VTA library: %s", dll_path)
        return runtime_dll[0]

    @tvm.register_func("tvm.rpc.server.load_module", override=True)
    def load_module(file_name):
        load_vta_dll()
        return _load_module(file_name)

    @tvm.register_func("device_api.ext_dev")
    def ext_dev_callback():
        load_vta_dll()
        return tvm.get_global_func("device_api.ext_dev")()

    @tvm.register_func("tvm.contrib.vta.init", override=True)
    def program_fpga(file_name):
        # pylint: disable=import-outside-toplevel
        env = get_env()
        if env.TARGET == "pynq":
            from pynq import xlnk

            # Reset xilinx driver
            xlnk.Xlnk().xlnk_reset()
        elif env.TARGET == "de10nano":
            # Load the de10nano program function.
            load_vta_dll()
        path = tvm.get_global_func("tvm.rpc.server.workpath")(file_name)
        bitstream_program(env.TARGET, path)
        logging.info("Program FPGA with %s ", file_name)

    @tvm.register_func("tvm.rpc.server.shutdown", override=True)
    def server_shutdown():
        if runtime_dll:
            runtime_dll[0].VTARuntimeShutdown()
            runtime_dll.pop()

    @tvm.register_func("tvm.contrib.vta.reconfig_runtime", override=True)
    def reconfig_runtime(cfg_json):
        """Rebuild and reload runtime with new configuration.

        Parameters
        ----------
        cfg_json : str
            JSON string used for configurations.
        """
        env = get_env()
        if runtime_dll:
            if env.TARGET == "de10nano":
                print("Please reconfigure the runtime AFTER programming a bitstream.")
            raise RuntimeError(
                "Can only reconfig in the beginning of session...")
        cfg = json.loads(cfg_json)
        cfg["TARGET"] = env.TARGET
        pkg = pkg_config(cfg)
        # check if the configuration is already the same
        if os.path.isfile(cfg_path):
            old_cfg = json.loads(open(cfg_path, "r").read())
            if pkg.same_config(old_cfg):
                logging.info("Skip reconfig_runtime due to same config.")
                return
        cflags = ["-O2", "-std=c++14"]
        cflags += pkg.cflags
        ldflags = pkg.ldflags
        lib_name = dll_path
        source = pkg.lib_source
        logging.info(
            "Rebuild runtime:\n output=%s,\n cflags=%s,\n source=%s,\n ldflags=%s",
            dll_path,
            "\n\t".join(cflags),
            "\n\t".join(source),
            "\n\t".join(ldflags),
        )
        cc.create_shared(lib_name, source, cflags + ldflags)
        with open(cfg_path, "w") as outputfile:
            outputfile.write(pkg.cfg_json)

    # program_fpga.__name__
