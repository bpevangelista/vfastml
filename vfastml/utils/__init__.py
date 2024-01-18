DEFAULT_API_SERVER_RPC_PORT = 9062
DEFAULT_MODEL0_SERVER_RPC_PORT = 9063

def _is_package_available(package_name: str, min_version: str = None) -> bool:
    import importlib.metadata
    from packaging.version import Version
    try:
        package_version = importlib.metadata.version(package_name)
        if min_version is None:
            return True
        else:
            return Version(package_version) >= Version(min_version)

    except importlib.metadata.PackageNotFoundError:
        return False

def print_model_parameters(model):
    total_param = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        total_param += param.numel()
        trainable_params += param.numel() if param.requires_grad else 0
    print(f'{{total_params: {total_param:,}, trainable_params: {trainable_params:,}}}')
