from langchain_community.llms import LlamaCpp
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class Loadllm:
    """
    uv pip install --force-reinstall --no-cache-dir `
    --index-url https://pypi.org/simple `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan `
    llama-cpp-python
    """
    @staticmethod
    def load_llm():

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path="./yi-chat-6B-GGUF/Yi-1.5-6B-Chat-Q4_K_M.gguf",
            # Offload as many layers as possible to the RTX 3060.
            n_gpu_layers=-1,
            n_batch=256,
            n_ctx=4096,
            callback_manager=callback_manager,
            verbose=True,
        )
        return llm
