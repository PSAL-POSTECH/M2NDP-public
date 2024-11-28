import argparse
import sys
import os
from benchmarks.histogram import *
from benchmarks.pagerank import *
from benchmarks.sssp import *
from benchmarks.vector_add import VectorAddKernel
# from benchmarks.layerNorm32.kernel0 import LayerNormKernel0
# from benchmarks.layerNorm32.kernel1 import LayerNormKernel1
# from benchmarks.naive_bayes.kernel0 import NaiveBayesKernel0
from benchmarks.dlrm import *
from benchmarks.imdb_gteq_lt_INT64 import GtEqLtINT64Kernel
from benchmarks.imdb_lt_INT64 import LtINT64Kernel
from benchmarks.imdb_gt_lt_FP32 import GtLtFP32Kernel
from benchmarks.imdb_two_col_AND import TwoColANDKernel
from benchmarks.imdb_three_col_AND import ThreeColANDKernel
# from benchmarks.kmeans import *
# from benchmarks.exponent import VectorExponentKernel
# from benchmarks.gelu import GeLUKernel
# from benchmarks.softmax import *
# from benchmarks.narrow_wide import NarrowKernel, WideKernel
# from benchmarks.gemv import *
from benchmarks.memset import *
from benchmarks.memcpy import *
from benchmarks.spmv import *
from utils.utils import *

packet_size = 32
data_size = 2

def get_kernel(kernel_name, num_m2ndps=1, arg=-1) :
    kernel = getattr(sys.modules[__name__], kernel_name)
    if arg != -1:
        return kernel(arg)
    else:
        return kernel()
    return kernel()

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, required=True, help='kernel name')
parser.add_argument('--output_dir', type=str, default='./tmp', help='output directory')
parser.add_argument('--run_cuda', action='store_true', help='run cuda', default=False)
parser.add_argument('--skip_functional_sim', action='store_true', help='skip functional simulation', default=False)
parser.add_argument('--config', type=str, default='m2ndp.config', help='config file')
parser.add_argument('--num_m2ndps', type=int, default=1, help='number of m2ndps')
parser.add_argument('--arg', type=float, default=-1, help='arg for kernel')

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.run_cuda:
        # Run cuda code for input, output generation
        if 'pagerank' in args.kernel :
            file_name = os.path.join(os.path.dirname(__file__), './cuda/pagerank/build_and_run.sh')
            os.system(f'bash {file_name}')
        if 'sssp' in args.kernel:
            file_name = os.path.join(os.path.dirname(__file__), './cuda/sssp/build_and_run.sh')
            os.system(f'bash {file_name}')

    kernel = get_kernel(args.kernel, args.num_m2ndps, args.arg)
    kernel_code = kernel.make_kernel()
    input_map = []
    output_map = []
    kernel_info = []
    if args.num_m2ndps == 1:
        input_map.append(kernel.make_input_map())
        kernel_info.append(kernel.get_kernel_info())
        output_map.append(kernel.make_output_map())
        make_input_files(kernel.kernel_name, kernel_code, input_map[0], 
                    output_map[0], kernel_info[0], file_dir=os.path.join(args.output_dir, str(0)))
    else:
        kernel_info = kernel.get_kernel_info(args.num_m2ndps)
        for i in range(args.num_m2ndps):
            input_map.append(kernel.make_input_map(i))
            output_map.append(kernel.make_output_map(i))
        for i in range(args.num_m2ndps):
            make_input_files(kernel.kernel_name, kernel_code, input_map[i], 
                            output_map[i], kernel_info[i], file_dir=os.path.join(args.output_dir, str(i)))
    if not args.skip_functional_sim:
        for i in range(args.num_m2ndps):
            execute_functional_sim(kernel.kernel_name, config=args.config, 
                                   file_dir=os.path.join(args.output_dir, str(i)))
