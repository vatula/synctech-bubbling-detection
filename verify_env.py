import torch
import torch.nn as nn

from src.utils.logger import get_logger, setup_project

# Import migraphx if available, otherwise define a stub
try:
    import migraphx  # type: ignore
except ImportError:
    migraphx = None

setup_project()
log = get_logger("verify_env")


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def verify() -> None:
    """Verifies the ROCm/MIGraphX environment."""
    log.info("Starting environment verification")

    # CUDA/ROCm Check
    cuda_available = torch.cuda.is_available()
    log.info("Checking CUDA/ROCm visibility", available=cuda_available)

    if not cuda_available:
        log.error(
            "CUDA/ROCm not found! Ensure ROCm drivers and PyTorch are "
            "correctly installed."
        )
    else:
        log.info("GPU details", device_name=torch.cuda.get_device_name(0))

    # Dummy Model Export
    log.info("Creating dummy model for ONNX export")
    model = DummyModel().eval()
    dummy_input = torch.randn(1, 10)
    onnx_path = "dummy_model.onnx"

    log.info("Exporting to ONNX", path=onnx_path)
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # MIGraphX Check
    if migraphx is None:
        log.warning("migraphx module not found. Skipping MIGraphX compilation check.")
    else:
        log.info("MIGraphX found. Attempting dummy compilation.")
        try:
            # Load ONNX and compile
            prog = migraphx.parse_onnx(onnx_path)
            prog.compile(migraphx.get_target("gpu"))
            log.info("MIGraphX dummy compilation: SUCCESS")
        except Exception as e:
            log.exception("MIGraphX compilation failed", error=str(e))

    log.info("Environment verification script completed")


if __name__ == "__main__":
    verify()
