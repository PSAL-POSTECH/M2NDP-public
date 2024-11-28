import argparse
import sys

from benchmarks.opt.fc import *
from benchmarks.opt.activation import *
from benchmarks.opt.layernom import *
from benchmarks.opt.residual import *
from benchmarks.opt.attention import *
from benchmarks.opt.allreduce import *
from utils.utils import *

packet_size = 32
data_size = 2

def get_kernel(kernel_name) :
    kernel = getattr(sys.modules[__name__], kernel_name)
    return kernel()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='kernel name')
parser.add_argument('--output_dir', type=str, default='./tmp', help='output directory')
parser.add_argument('--context_len', type=int, default=1024, help='context length')
parser.add_argument('--skip_functional_sim', action='store_true', help='skip functional simulation', default=False)
parser.add_argument('--config', type=str, default='m2ndp.config', help='config file')
parser.add_argument('--model_parallelism', type=int, default=1, help='config file')

if __name__ == "__main__":
    args = parser.parse_args()

    kernel_map = {
        'layernorm_0': OptLayerNorm0(args.model),
        'layernorm_1': OptLayerNorm1(args.model),
        'qkv_proj':  OptQKVProj(args.model, args.model_parallelism),
        'fc1': OptFc1(args.model, args.model_parallelism),
        'relu': OptRelu(args.model),
        'fc2': OptFc2(args.model, args.model_parallelism),
        'out_proj': OptOutProj(args.model, args.model_parallelism),
        'residual': OptResidual(args.model),
        'attention1': OptAttention1Kernel(args.model, args.context_len, 2048, args.model_parallelism),
        'attention2': OptAttention2Kernel(args.model, args.context_len, args.model_parallelism),
    }

    for kernel_name, kernel in kernel_map.items():
        kernel_code = kernel.make_kernel()
        input_map = kernel.make_input_map()
        output_map = kernel.make_output_map()
        kernel_info = kernel.get_kernel_info()
        output_dir = args.output_dir + '/' + kernel_name
        make_input_files(kernel.kernel_name, kernel_code, input_map, output_map, kernel_info, file_dir=output_dir)
        if not args.skip_functional_sim:
            execute_functional_sim(kernel.kernel_name, config=args.config, file_dir=output_dir)
    if args.model_parallelism > 1:
        reduce_scatter = OptReduceScatter(args.model, args.model_parallelism)
        kernel_code = reduce_scatter.make_kernel()
        input_map = reduce_scatter.make_input_map()
        output_map = reduce_scatter.make_output_map()
        kernel_info = reduce_scatter.get_kernel_info(num_m2ndps=args.model_parallelism)
        output_dir = args.output_dir + '/' + reduce_scatter.kernel_name
        make_input_files_multi_ndp(reduce_scatter.kernel_name, kernel_code, input_map, output_map, kernel_info, file_dir=output_dir, model_parallelism=args.model_parallelism)
        if not args.skip_functional_sim:
            for i in range(args.model_parallelism):
                execute_functional_sim(reduce_scatter.kernel_name, config=args.config, file_dir=output_dir + f'/{i}')

        all_gather = OptAllGather(args.model, args.model_parallelism)
        kernel_code = all_gather.make_kernel()
        input_map = all_gather.make_input_map()
        output_map = all_gather.make_output_map()
        kernel_info = all_gather.get_kernel_info(num_m2ndps=args.model_parallelism)
        output_dir = args.output_dir + '/' + all_gather.kernel_name
        make_input_files_multi_ndp(all_gather.kernel_name, kernel_code, input_map, output_map, kernel_info, file_dir=output_dir, model_parallelism=args.model_parallelism)
        if not args.skip_functional_sim:
            for i in range(args.model_parallelism):
                execute_functional_sim(all_gather.kernel_name, config=args.config, file_dir=output_dir + f'/{i}')