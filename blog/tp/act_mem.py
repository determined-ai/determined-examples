from typing import Any, Iterable, Optional, Union

import torch


def B_to_GiB(bytes: Union[int, float]) -> float:
    return bytes / 2**30


def get_tensor_bytes(tensor: torch.Tensor) -> int:
    """
    Returns the bytes of storage a given tensor takes up. If `tensor` is a view of a larger tensor,
    this function only returns the bytes associated with the view.
    """
    tensor_bytes = tensor.numel() * tensor.element_size()
    return tensor_bytes


class AllocatedMemContext:
    """
    Context manager which captures the allocated GPU memory at context exit and the change between
    enter and exit.

    Only includes `allocated_bytes.all.`-prefixed keys in `memory_stats` with all readings converted
    to GiB.

    Example:

    ```python

    ```
    """

    def __init__(self) -> None:
        # Ensure CUDA libraries are loaded:
        torch.cuda.current_blas_handle()

        self.before: dict[str, int] = {}
        self.after: dict[str, int] = {}
        self.delta: dict[str, int] = {}

        self._mem_key_prefix = "allocated_bytes.all."

    def _get_mem_dict(self) -> dict[str, int]:
        return {
            k.replace(self._mem_key_prefix, ""): v
            for k, v in torch.cuda.memory_stats().items()
            if self._mem_key_prefix in k
        }

    def __enter__(self) -> "AllocatedMemContext":
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}


class SavedTensorContext:
    """
    Context manager which captures all tensors which are registered as being saved for backwards
    within the context window. Does not work with `meta`-device tensors.

    All saved tensors are stored in the `saved_tensor_dict` attr, which is an instance of torch's
    WeakTensorKeyDictionary with tensor/data_ptr key/value pairs. Some of these tensors may be
    views of the same underlying storage. The total memory of all saved tensors in bytes, accounting
    for redundant views, can be accessed through `saved_tensor_mem`.

    Use:
    ```
    model = ...
    with SavedTensorContext(ignored_tensors=model.parameters()) as saved:
        # Do some computation with `model` and capture saved tensors which are not model weights

    ```
    saved.saved_tensor_dict # WeakTensorKeyDictionary of all saved tensors.
    saved.saved_tensor_mem # bytes from all saved tensors (activation memory).
    """

    def __init__(
        self,
        ignored_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> None:
        # Track ignored tensors by their storage's data_ptr. Important to use storage's data_ptr,
        # not just the data_ptr of the tensor itself.
        self._ignored_data_ptrs = (
            set()
            if ignored_tensors is None
            else {t.untyped_storage().data_ptr() for t in ignored_tensors}
        )

        # Use WeakTensorKeyDictionary instances to save non-trivial tensor references, since these
        # won't keep the tensor alive if the only references to the tensor are within this object.
        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = saved_tensor.untyped_storage().data_ptr()
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[saved_tensor] = data_ptr
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)

    def __enter__(self) -> "SavedTensorContext":
        self._saved_tensors_hook.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_tensors_hook.__exit__(*args, **kwargs)

    @property
    def saved_tensor_mem(self) -> int:
        """
        The memory in bytes of all saved tensors, accounting for views into the same storage.
        """
        accounted_for = self._ignored_data_ptrs.copy()
        total_bytes = 0
        for t in self.saved_tensor_dict:
            data_ptr = t.untyped_storage().data_ptr()
            if data_ptr not in accounted_for:
                total_bytes += t.untyped_storage().nbytes()
                accounted_for.add(data_ptr)
        return total_bytes
