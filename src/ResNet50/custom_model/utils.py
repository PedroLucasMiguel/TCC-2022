"""
A implementação presente neste arquvivo é basicamente a mesma encontrada aqui:
https://github.com/pytorch/vision/blob/master/torchvision/models/utils.py
"""
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
