# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain

'''#deepspeed deps - remove soon:
import deepspeed
from deepspeed.launcher.runner import main
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
from megatron.neox_arguments import NeoXArgs
#from megatron.utils import get_wandb_api_key
neox_args = NeoXArgs.consume_deepy_args()
deepspeed_main_args = neox_args.get_deepspeed_main_args()
'''

# python ./deepy.py train.py -d configs small.yml local_setup.yml
# python fsdp_train.py -d configs small.yml local_setup.yml


def environ_check():
    print("\n--> gpu architecture")
    print(torch.cuda.get_arch_list())

    print("\n--> GPU 0 specifics:")

    print(torch.cuda.get_device_properties(torch.device('cuda')))

    print("\n--> Total devices:")
    print(torch.cuda.device_count())


neox_args = NeoXArgs.consume_fsdp_args()
deepspeed_main_args = neox_args.get_deepspeed_main_args()
megatronconfig = deepspeed_main_args[deepspeed_main_args.index("--megatron_config")+1]

print("@@@@@@@@@@@@@@@@@@@", deepspeed_main_args)
neox_args.consume_neox_args(megatronconfig=megatronconfig)
neox_args.configure_distributed_args()
neox_args.build_tokenizer() 
neox_args.initialize_tensorboard_writer() 
pretrain(neox_args=neox_args)
# def main():
#     environ_check()
#     # neox_args = NeoXArgs.consume_deepspeed_args()
#     # deepspeed_main_args = neox_args.get_deepspeed_main_args()
#     megatronconfig = deepspeed_main_args[deepspeed_main_args.index("--megatron_config")+1]
#     #main(deepspeed_main_args)
#     # neox_args = NeoXArgs.consume_deepy_args()
    
#     neox_args.consume_neox_args()
#     neox_args.configure_distributed_args()
#     neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
#     neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
#     pretrain(neox_args=neox_args)




# if __name__ == "__main__":
#     main()

