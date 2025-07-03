import torch
from detectron2.modeling import build_model

def build_compiled_model(cfg):
    normal_model = build_model(cfg)

    if cfg.MODEL.COMPILE.ENABLED:
        args = {}
        options = {}
        if cfg.MODEL.COMPILE.TRACE:
            options["trace.enabled"] = cfg.MODEL.COMPILE.TRACE
        if cfg.MODEL.COMPILE.MODE != "default":
            args["mode"] = cfg.MODEL.COMPILE.MODE
        
        # Options arguments can't be set with other parameters. 
        # So only add it if needed.
        if options:
            args["options"] = options
        compiled_model = torch.compile(normal_model, **args)
    else:
        compiled_model = normal_model
    
    return normal_model, compiled_model
