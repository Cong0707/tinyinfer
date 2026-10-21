"""tinyinfer: impossible inference on tiny hardware.

复用 DeepInfer 的算法与结构定义，在消费级硬件上直接加载 HF 权重。
"""
import os as _os

# Streaming layers churn 190MB device tensors per token; the default caching allocator
# fragments badly on an 8GB card (measured: +14% decode with expandable segments).
# Must be set before torch is first imported, hence the package __init__.
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
