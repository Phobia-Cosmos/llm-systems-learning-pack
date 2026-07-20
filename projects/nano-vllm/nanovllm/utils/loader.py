import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # 问题（已回答）：packed_modules_mapping 在做什么，里面存储什么？
    # 回答：它描述 checkpoint 中的独立参数如何装入模型里的融合参数，键是源参数名片段，值是
    # “目标参数名片段、逻辑分片 id”；例如 q_proj -> (qkv_proj, "q")，本身不存权重 Tensor。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        # 问题（已回答）：safe_open 的 "pt" 和第三个参数 "cpu" 分别有什么作用？
        # 回答："pt" 要求以 PyTorch Tensor 形式读取，"cpu" 指定 Tensor 的加载设备。上下文管理器按需读取
        # safetensors 文件并在退出时释放句柄，避免先把整个 checkpoint 一次性搬到 GPU。
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 问题（已回答）：v 和 shard_id 分别做什么？
                        # 回答：v 是模型中融合模块的名称片段；shard_id 标识源权重属于融合参数的哪一段，
                        # 例如 "q"、"k"、"v"，供该 Parameter 的自定义 weight_loader 定位目标切片。
                        v, shard_id = packed_modules_mapping[k]
                        # 问题（已回答）：为什么用 replace 得到 param_name？
                        # 回答：checkpoint 名称仍指向独立模块，模型却注册了融合模块；替换名称片段后才能用
                        # get_parameter 找到真实目标 Parameter，再把源 Tensor 写入 shard_id 对应的区域。
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
