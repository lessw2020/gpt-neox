
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




def main():
    environ_check()
    #main(deepspeed_main_args)
    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    pretrain(neox_args=neox_args)




if __name__ == "__main__":
    main()
