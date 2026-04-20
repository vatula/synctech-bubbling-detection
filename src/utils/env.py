import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


def setup_env() -> None:
    """Configures the environment for the project, including hardware optimizations."""
    if torch.cuda.is_available():
        # Addressing PyTorch warning for AMD/NVIDIA Tensor/Matrix Cores
        # Trade-off precision for performance
        torch.set_float32_matmul_precision("high")
        log.info(
            "Float32 matmul precision set to 'high'",
            device=torch.cuda.get_device_name(0),
        )
    else:
        log.info("CUDA not available, skipping hardware-specific optimizations")
