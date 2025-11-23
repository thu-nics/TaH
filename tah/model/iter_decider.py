import os
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Any, Type, Union, Tuple, List

from tah.model.registry import (
    register_iter_decider,
    get_iter_decider_class,
    capture_init_args,
    mark_wrapper_iter_decider,
)


POSITIVE_INFINITY_LOGITS = 10.0
MINUS_INFINITY_LOGITS = -10.0
NEUTRAL_LOGITS = 0.0

class IterDecider(nn.Module):
    """Base class for deciding whether to continue iterating a token.

    All IterDecider implementations must efficiently handle inputs of arbitrary shape (..., vocab_size)
    where (...) can be any number of leading dimensions (batch, sequence, etc.).
    """

    def __init__(self, threshold: float = 0.5, max_iter: int = 3):
        super().__init__()
        # store as buffer to allow assignment on subclasses without property conflicts
        self.register_buffer("threshold", torch.tensor(float(threshold), dtype=torch.float32))
        self.max_iter = max_iter

    def forward(self, logits: torch.Tensor, iter_depth: int, **kwargs) -> torch.Tensor:
        """
        Decide whether to continue iterating a token.

        Args:
            logits: The logits of the token, shape (..., vocab_size) where (...)
                   represents arbitrary leading dimensions
            iter_depth: The iteration depth of the token that has been processed.
            Optional kwargs:
                - hidden_states: The hidden states of the token, shape (..., hidden_size) where (...)

        Returns:
            A float tensor of shape (...) with values between 0 and 1,
            indicating the probability of continuing iteration.
            The output preserves all leading dimensions from the input.
        """
        raise NotImplementedError

@register_iter_decider
@capture_init_args
class TrivialIterDecider(IterDecider):
    """Trivial iteration decider that always ends.

    Efficiently handles arbitrary input shapes (..., vocab_size) by returning
    a boolean tensor of shape (...,) filled with False values.
    """

    def __init__(self, max_iter: int = 1):
        super().__init__(max_iter=max_iter)

    def forward(self, logits: torch.Tensor, iter_depth: int, **kwargs) -> torch.Tensor:
        decision = torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device)
        logits_out = torch.full(decision.shape, NEUTRAL_LOGITS, dtype=logits.dtype, device=logits.device)
        return decision, logits_out

@register_iter_decider
@capture_init_args
class IterLabelDecider(IterDecider):
    """Iteration decider that strictly follows provided iter_count_labels.

    Decision rule: continue if and only if (iter_count_labels > iter_depth) for valid tokens.
    padding/ignored tokens (-100) will always stop.
    """

    def __init__(self, max_iter: int = 3):
        super().__init__(max_iter=max_iter)

    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        iter_count_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (iter_depth >= self.max_iter):
            decision = torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device)
            logits_out = torch.full(decision.shape, MINUS_INFINITY_LOGITS, dtype=logits.dtype, device=logits.device)
            return decision, logits_out
        if (iter_count_labels is None):
            decision = torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device)
            logits_out = torch.full(decision.shape, NEUTRAL_LOGITS, dtype=logits.dtype, device=logits.device)
            return decision, logits_out

        valid_mask = (iter_count_labels != -100)
        decision_bool = (iter_count_labels > iter_depth) & valid_mask
        decision = decision_bool
        logits_out = torch.full(decision.shape, NEUTRAL_LOGITS, dtype=logits.dtype, device=logits.device)
        return decision, logits_out

class ClassifierBlock(nn.Module):
    """
    A single transformer-style block for the classifier backbone.
    Implements layer normalization, MLP with expansion, and residual connections.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        expansion_factor=4,
        dropout_rate=0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_norm = nn.LayerNorm(input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * expansion_factor, output_dim),
            nn.Dropout(dropout_rate),
        )

        self.dim_change = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        normalized = self.layer_norm(x)
        residual = self.dim_change(x)
        return residual + self.mlp(normalized)


class ClassifierBackbone(nn.Module):
    """
    Backbone architecture for all classifiers.
    Implements transformer-style MLP blocks with residual connections.
    Position embeddings are disabled in this setup.
    """

    def __init__(
        self,
        input_dim,
        output_dim=1,
        hidden_dims=[256, 512, 256],
        expansion_factor=4,
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
    ):
        super().__init__()
        self.use_position_embedding = use_position_embedding

        self.blocks = nn.ModuleList()

        self.input_projection = nn.Linear(input_dim, hidden_dims[0])

        block_dims = hidden_dims + [hidden_dims[-1]]

        for i in range(len(block_dims) - 1):
            block_input_dim = block_dims[i]
            block_output_dim = block_dims[i + 1]
            self.blocks.append(
                ClassifierBlock(
                    input_dim=block_input_dim,
                    output_dim=block_output_dim,
                    expansion_factor=expansion_factor,
                    dropout_rate=dropout_rate,
                )
            )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def apply_position_embedding(self, x, position_ids=None):
        # No position embeddings used in this setup
        return x

    def forward(self, x, position_ids=None):
        x = self.input_projection(x)
        x = self.apply_position_embedding(x, position_ids)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

@register_iter_decider
@capture_init_args
class MLPIterDecider(IterDecider):
    """Classifier-based iteration decider using hidden states and top-k logits."""

    def __init__(
        self,
        topk: int = 100,
        hidden_states_size: int = 1024,
        hidden_states_layer_nums: list = [16,20,24,28], # explicit layer indices to use from all_hidden_states
        hidden_dims: list = [256, 512, 256],
        expansion_factor: int = 4,
        dropout_rate: float = 0.3,
        normalize_input: bool = False,
        threshold: float = 0.5,
        max_iter: int = 3,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(max_iter=max_iter)
        self.topk = topk
        self.hidden_states_size = hidden_states_size
        self.hidden_states_layer_nums = list(hidden_states_layer_nums)
        if hasattr(self.__class__, 'threshold'):
            delattr(self, 'threshold')
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=dtype, requires_grad=True))
        self.max_iter = max_iter

        self.normalize_input = normalize_input
        if self.normalize_input:
            num_selected = max(1, len(self.hidden_states_layer_nums))
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size * num_selected)

        # Project top-k logits to hidden state size
        self.logits_projection = nn.Linear(self.topk, hidden_states_size, dtype=dtype)

        # Combine hidden states and projected logits
        num_selected = max(1, len(self.hidden_states_layer_nums))
        combined_size = hidden_states_size * num_selected + hidden_states_size
        self.combined_projection = nn.Linear(combined_size, hidden_dims[0], dtype=dtype)

        # Backbone MLP stack
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=1,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, logits: torch.Tensor, iter_depth: int, all_hidden_states: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if iter_depth >= self.max_iter:
            decision = torch.zeros(
                logits.shape[:-1], dtype=torch.bool, device=logits.device
            )
            logits_out = torch.full(decision.shape, MINUS_INFINITY_LOGITS, dtype=logits.dtype, device=logits.device)
            return decision, logits_out

        original_shape = logits.shape[:-1]

        # Build hidden features from requested layers without padding
        num_selected = max(1, len(self.hidden_states_layer_nums))
        if all_hidden_states is None:
            # Fallback to zeros if hidden states are unavailable
            hidden_concat = torch.zeros(*original_shape, self.hidden_states_size * num_selected, device=logits.device, dtype=logits.dtype)
        else:
            hs = all_hidden_states
            # Expect shape (..., L, H); if (..., H) provided, treat as single-layer
            if hs.dim() == logits.dim():
                hs = hs.unsqueeze(-2)
            total_layers = hs.size(-2)
            if num_selected == 1 and len(self.hidden_states_layer_nums) == 0:
                indices = [total_layers - 1]
            else:
                indices = self.hidden_states_layer_nums
            index_tensor = torch.as_tensor(indices, device=hs.device, dtype=torch.long)
            if index_tensor.numel() == 0:
                raise ValueError("hidden_states_layer_nums must not be empty")
            if torch.min(index_tensor).item() < 0 or torch.max(index_tensor).item() >= total_layers:
                raise ValueError(f"hidden_states_layer_nums out of range: {indices}, total_layers={total_layers}")
            selected = torch.index_select(hs, dim=-2, index=index_tensor)  # (..., num_selected, H)
            hidden_concat = selected.reshape(*original_shape, selected.size(-2) * self.hidden_states_size)

        # Mirror PluginNeuralIterDecider behavior: apply top-k on logits
        k = min(self.topk, logits.size(-1))
        topk_values, _ = torch.topk(logits, k=k, dim=-1)

        # Optional normalization
        if self.normalize_input:
            hidden_concat = self.layer_norm_hidden_states(hidden_concat)
            topk_values = torch.softmax(topk_values, dim=-1)

        # Project logits and combine
        logits_features = self.logits_projection(topk_values)
        combined_features = torch.cat([hidden_concat, logits_features], dim=-1)
        x = self.combined_projection(combined_features)

        decision_logits = self.backbone(x)
        if decision_logits.dim() == logits.dim():
            decision_logits = decision_logits.squeeze(-1)

        decision_scores = self.sigmoid(decision_logits)
        thr = self.threshold
        if isinstance(thr, torch.Tensor):
            thr = float(thr.detach().item())
        decision_mask = (decision_scores > thr)
        return decision_mask, decision_logits


@register_iter_decider
@capture_init_args
class AlwaysWrapperIterDecider(IterDecider):
    """Wrapper that enforces a simple control-flow policy around a base iter decider.

    Modes:
    - "continue": force continuing until the final allowed iteration (previous behavior)
    - "stop": stop after the first iteration

    Finishing rule via threshold (used by TaH's finished_mask = (prob <= threshold)):
    - continue: threshold = -1.0 until last iteration, then 1.0 to finish all
    - stop: threshold = 1.0 at the first iteration so all tokens finish immediately
    """

    def __init__(
        self,
        max_iter: int = 3,
        base_iter_decider_cls: str = "MLPIterDecider",
        base_iter_decider_kwargs: Optional[dict] = None,
        mode: str = "continue",
    ):
        super().__init__(max_iter=max_iter)
        if not isinstance(base_iter_decider_cls, str):
            raise ValueError("AlwaysWrapperIterDecider expects base_iter_decider_cls as a string class name")
        mode = str(mode).lower().strip()
        if mode not in ("continue", "stop"):
            raise ValueError("AlwaysWrapperIterDecider mode must be either 'continue' or 'stop'")
        self.mode = mode

        base_cls = get_iter_decider_class(base_iter_decider_cls)
        base_iter_decider_kwargs = dict(base_iter_decider_kwargs or {})
        base_iter_decider_kwargs.setdefault("max_iter", max_iter)
        self.base_iter_decider = base_cls(**base_iter_decider_kwargs)
        self._last_forward_iter_depth: Optional[int] = None

    def update_training_state(self, current_step: int, current_epoch: int):
        if getattr(self, 'base_iter_decider', None) is not None and hasattr(self.base_iter_decider, 'update_training_state') and callable(self.base_iter_decider.update_training_state):
            try:
                self.base_iter_decider.update_training_state(current_step=current_step, current_epoch=current_epoch)
            except Exception:
                pass

    def forward(self, logits: torch.Tensor, iter_depth: int, **kwargs) -> torch.Tensor:
        # Respect cap for shape consistency
        if iter_depth >= self.max_iter:
            decision = torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device)
            logits_out = torch.full(decision.shape, MINUS_INFINITY_LOGITS, dtype=logits.dtype, device=logits.device)
            return decision, logits_out

        self._last_forward_iter_depth = int(iter_depth)

        # Delegate to base decider to obtain its logits if any
        base_output = self.base_iter_decider(logits, iter_depth, **kwargs)
        if (not isinstance(base_output, tuple)) or (len(base_output) < 2):
            raise TypeError("Base iter decider must return a (decision_bool, logits) tuple")
        base_decision, base_logits = base_output[0], base_output[1]

        if self.mode == "continue":
            decision = torch.ones_like(base_decision, dtype=torch.bool, device=base_decision.device)
        else:  # stop after first iteration
            decision = torch.zeros_like(base_decision, dtype=torch.bool, device=base_decision.device)
        return decision, (base_logits.to(dtype=logits.dtype) if base_logits is not None else torch.full_like(decision, NEUTRAL_LOGITS, dtype=logits.dtype))

    @property
    def threshold(self) -> float:
        # If last forward depth is unknown, fall back to base threshold or 0.5
        if self._last_forward_iter_depth is None:
            thr = getattr(self.base_iter_decider, 'threshold', None)
            try:
                if isinstance(thr, torch.Tensor):
                    return float(thr.detach().item())
            except Exception:
                pass
            return float(thr) if thr is not None else 0.5

        if self.mode == "continue":
            # Continue until final iteration
            if self._last_forward_iter_depth < self.max_iter:
                return -1.0
            else:
                return 1.0
        else:
            # Stop mode: finish immediately at the first iteration
            return 1.0


@register_iter_decider
@capture_init_args
class OracleDynamicIterDecider(IterDecider):
    """
    Use LLM forward in the first iter to get the oracle token.
    In subsequent iters, compare the predicted token with the oracle token.
    """

    def __init__(
        self,
        max_iter: int = 3,
        ref_model_path: Optional[str] = None,
        dtype: Union[torch.dtype, str] = "auto",
        device: Optional[str] = None,
        use_kv_cache: bool = True,
        backend: str = 'sglang',
        false_positive_rate: float = 0,
        false_negative_rate: float = 0
    ):
        '''
        Note: the ref_model is default to hf model
        Args:
        - max_iter (int): The maximum number of iterations to perform.
        - ref_model_path (str, optional): The path to the reference model.
        - dtype (torch.dtype or str, optional): The data type to use for the model.
        - device (str, optional): The device to run the model on.
        - use_kv_cache (bool, optional): Whether to use key-value caching.
        - backend (str, optional): The backend to use for the model. options: ['sglang', 'hf']
        '''
        super().__init__()
        self.max_iter = max_iter
        self.use_kv_cache = use_kv_cache
        self._dtype = dtype
        self._device = device
        self.backend = backend

        self.ref_model = None
        self._ref_model_path = ref_model_path
        print('oracle iterdecider ref_model path:', self._ref_model_path)
        self._ref_past = None
        # Cache of oracle tokens from reference model prefill
        # For HF backend prefill at iter_depth==1, we cache per-position greedy tokens
        # shape: (batch, seq_len)
        self._cached_tokens_full = None
        self._last_step_depth = None

        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate


    @torch.no_grad()
    def _ensure_ref(self, ref_model):
        if ref_model is not None:
           return ref_model
        if self.ref_model is not None:
            return self.ref_model
        if self._ref_model_path is None:
            raise RuntimeError(
                "OracleDynamicIterDecider needs a reference model: please provide ref_model_path in init "
                "or pass ref_model in forward()."
            )
        if self.backend == 'hf':
            torch_dtype = None
            if isinstance(self._dtype, str) and self._dtype != "auto":
                torch_dtype = getattr(torch, self._dtype)
            elif isinstance(self._dtype, torch.dtype):
                torch_dtype = self._dtype
            from transformers import AutoModelForCausalLM
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self._ref_model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if self._device is None else None,
                attn_implementation='sdpa',
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self._device is not None:
                self.ref_model.to(self._device)
            self.ref_model.eval()
            return self.ref_model
        elif self.backend == 'sglang':
            from transformers import AutoTokenizer, AutoConfig
            import socket, os
            import multiprocessing as mp

            class _SGLGreedyOracle:
                """

                """
                def __init__(self, model_path: str, dtype="auto"):
                    print('Initializing sgl engine')

                    self.tok = AutoTokenizer.from_pretrained(
                        model_path, trust_remote_code=True, use_fast=True
                    )
                    self.pad_id = self.tok.pad_token_id
                    self.vocab_size = self.tok.vocab_size

                    def _to_sgl_dtype(dt):
                        if dt in (None, "auto"): return "auto"
                        if isinstance(dt, torch.dtype):
                            m = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
                            return m.get(dt, "auto")
                        if isinstance(dt, str): return dt
                        return "auto"

                    def _get_idx():
                        try:
                            name = mp.current_process().name
                            return int(name.split("-")[-1])
                        except Exception:
                            return 0

                    def _next_free_port(start):
                        p = start
                        while True:
                            with socket.socket() as s:
                                try:
                                    s.bind(("", p))
                                    return p
                                except OSError:
                                    p += 1

                    idx = _get_idx()

                    BASE = 32000 + idx * 20
                    http_port = _next_free_port(BASE + 0)   
                    rdzv_port = _next_free_port(BASE + 1)  
                    nccl_port = _next_free_port(BASE + 2)  

                    print(f'[Job {idx}] SGLang Engine ports: HTTP {http_port}, RDZV {rdzv_port}, NCCL {nccl_port}')

                    os.environ["SGLANG_PORT"] = str(http_port) 
                    os.environ["NCCL_IB_DISABLE"] = "1"         

                    os.environ["MASTER_ADDR"] = "127.0.0.1"
                    os.environ["MASTER_PORT"] = str(rdzv_port)
                    os.environ["SGLANG_PORT"] = str(http_port)
                    os.environ["NCCL_IB_DISABLE"] = "1"       

                    import sglang as sgl
                    self.eng = sgl.Engine(
                        model_path=model_path,
                        tokenizer_path=model_path,
                        trust_remote_code=True,
                        dtype=_to_sgl_dtype(dtype),
                        tp_size=2,
                        dp_size=1,
                        enable_dp_attention=False,
                        host='127.0.0.1',
                        port=http_port,                                # HTTP
                        dist_init_addr=f"127.0.0.1:{rdzv_port}",       # **For TCPStore port**
                    )
                    self._running_ids = None  # list[list[int]]


                def _trim_left_pad(self, ids: list[int]) -> list[int]:
                    if self.pad_id is None or not ids: return ids
                    start = 0
                    while start < len(ids) and ids[start] == self.pad_id:
                        start += 1
                    return ids[start:]
                
                def _tokenize_one(self, text: str) -> List[int]:
                    if not text:
                        return []
                    return self.tok(text, add_special_tokens=False, return_attention_mask=False).input_ids

                def greedy_next_token_ids(self, input_ids: torch.Tensor, fresh: bool) -> torch.Tensor:
                    assert isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2, \
                        "input_ids must be a 2D LongTensor (B, T)"
                    device = input_ids.device
                    ids_list = input_ids.tolist()
                    B = len(ids_list)

                    if fresh or self._running_ids is None:
                        # new sequence
                        ids_list = [self._trim_left_pad(x) for x in ids_list]
                        self._running_ids = [list(x) for x in ids_list]
                    else:
                        # continue
                        if len(self._running_ids) != B:
                            raise RuntimeError(
                                "fresh=False but batch size changed, set fresh=True when starting a new batch."
                            )
                        for i in range(B):
                            self._running_ids[i].extend(ids_list[i])

                    # allow <eos>
                    outs = self.eng.generate(
                        input_ids=self._running_ids,
                        sampling_params={
                            "max_new_tokens": 1,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "ignore_eos": False,   # allow <eos>
                        },
                        return_logprob=False,
                    )

                    next_ids: List[int] = []
                    for i, o in enumerate(outs):
                        nid: Optional[int] = None

                        # for different version
                        for k in ("token_ids", "output_ids"):
                            val = o.get(k, None)
                            if isinstance(val, list) and len(val) >= 1:
                                nid = int(val[0])
                                break

                        if nid is None:
                            cont = o.get("text", "")
                            new_ids = self._tokenize_one(cont)
                            if new_ids:
                                nid = int(new_ids[0])
                            else:
                                # if model did not return new_id, return eos
                                nid = int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else 0

                        next_ids.append(nid)
                        # self._running_ids[i].append(nid)
                    result = torch.tensor(next_ids, dtype=torch.long, device=device).detach()
                    del outs, ids_list, next_ids
                    return result

            config = AutoConfig.from_pretrained(self._ref_model_path, trust_remote_code=True)
            self.ref_model = _SGLGreedyOracle(self._ref_model_path, dtype=self._dtype)
            self.ref_model.config = config
            return self.ref_model
        else:
            raise RuntimeError("Unsupported backend: {}".format(self.backend))

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        iter_depth: int,
        active_valid_mask: torch.Tensor,
        prediction_logits: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        fresh: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        '''
        Args:
        - logits (torch.Tensor): The logits tensor from the model.
        - iter_depth (int): The current iteration depth.
        - active_valid_mask (torch.Tensor): shape (batch, max_active_len)
        - prediction_logits (torch.Tensor): next prediction logits from base model (batch, vocab)
        - input_ids (torch.Tensor, optional): (batch, seq)
        - ref_model (nn.Module, optional): The reference model to use.
        - fresh (torch.Tensor, optional): If True, reset the reference model state.
        - **kwargs: Additional keyword arguments.
        '''
        if iter_depth >= self.max_iter:
            return torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device), None

        # iter depth examination
        if self._last_step_depth is None and iter_depth > 1:
            raise RuntimeError("Cannot start OracleDynamicIterDecider with iter_step > 1")
        if self._last_step_depth is not None and iter_depth <= self._last_step_depth and iter_depth > 1:
            print("[OracleDynamicIterDecider] Warning: iter_depth not increasing, make sure it restarts from 1 per sequence.")
        self._last_step_depth = iter_depth

        # Note: For per-position decisions we prefer using the provided per-token logits (logits arg)
        # which corresponds to active positions flattened by active_valid_mask. We'll reconstruct
        # a (B, T) view later. prediction_logits (B, V) may reflect only the last position and is
        # insufficient for full prefill decisions.

        if iter_depth == 1:
            ref = self._ensure_ref(ref_model)
            if input_ids is None:
                raise RuntimeError("input_ids is needed when iter_depth == 1")            

            # assert prediction_logits.size(-1) == ref.config.vocab_size, \
            #      f"Base model vocab != ref model vocab(base {prediction_logits.size(-1)}, ref {ref.config.vocab_size}). Use same tokenizer/vocab for oracle comparison."

            batch_size, query_len = input_ids.shape
            # check if new sequence
            if fresh:
                self._ref_past = None
                self._cached_tokens_full = None

            if self.backend == 'hf':
                # Prepare attention mask aligned to current query length
                attention_mask = attention_mask[:, -query_len:] if attention_mask is not None else None
                outputs = ref(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=self._ref_past,
                )

                if self.use_kv_cache:
                    self._ref_past = outputs.past_key_values

                # Cache per-position oracle tokens for the whole sequence (B, T)
                # We only keep argmax tokens to avoid storing full logits
                self._cached_tokens_full = outputs.logits.argmax(dim=-1)
                # Free large logits tensor ASAP
                del outputs
            elif self.backend == 'sglang':
                # For sglang backend we only support next-token oracle for decode use-case.
                # To maintain compatibility, fill the last position using sglang's next token
                next_ids = ref.greedy_next_token_ids(input_ids, fresh=fresh)
                # Build a per-position cache with only the last position populated
                self._cached_tokens_full = torch.zeros_like(input_ids)
                self._cached_tokens_full[:, -1] = next_ids

        # Ensure oracle tokens are available
        if self._cached_tokens_full is None:
            raise RuntimeError("OracleDynamicIterDecider must have per-position oracle tokens cached at iter 1.")

        # Compute base predictions for all active positions from current iteration logits
        # logits shape: (num_valid_positions, vocab)
        base_pred_flat = logits.argmax(dim=-1)

        # Reconstruct per-position matrix (B, T) for base predictions
        # Fill only where active_valid_mask == 1
        base_pred_full = torch.zeros_like(input_ids)
        base_pred_full[active_valid_mask == 1] = base_pred_flat

        # Compute per-position continue decisions: continue if base iter1 != oracle
        continue_full = (base_pred_full != self._cached_tokens_full).to(torch.bool)
        # Only consider valid positions
        continue_full = continue_full & active_valid_mask.bool()

        # Apply noise per-position
        if self.false_negative_rate > 0 or self.false_positive_rate > 0:
            rand_fn = torch.rand_like(continue_full.float())
            rand_fp = torch.rand_like(continue_full.float())
            continue_full = (continue_full | (rand_fn < self.false_negative_rate)) & (rand_fp >= self.false_positive_rate)

        # Flatten back to match expected return shape (num_valid_positions,)
        continue_mask = continue_full[active_valid_mask == 1].bool().to(logits.device)

        return continue_mask, None


def save_iter_decider(iter_decider: IterDecider, save_directory: str):
    """Save iter_decider state dict and configuration."""
    # Use captured initialization arguments from the decorator
    init_args = getattr(iter_decider, "_init_args", {})

    # Use natural state_dict - no overrides needed
    state_dict = iter_decider.state_dict()
    state_dict = {k: v.cpu() for k, v in state_dict.items()}
    data = {
        "class": iter_decider.__class__.__name__,
        "state_dict": state_dict,
        "init_args": init_args,
    }

    save_path = os.path.join(save_directory, "iter_decider.bin")
    print(f"Saving iter_decider with {len(state_dict)} parameters to {save_path}")
    torch.save(data, save_path)


def load_iter_decider(load_directory: str, class_name: Optional[str] = None, init_args: Optional[dict] = None) -> IterDecider:
    """Load iter_decider from directory."""
    path = os.path.join(load_directory, "iter_decider.bin")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No iter_decider found at {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    if class_name is None:
        class_name = data.get("class")

    if not class_name:
        raise ValueError("No iter_decider class specified in saved data")

    # Get constructor arguments if available
    if init_args is None:
        init_args = data.get("init_args", {})

    # Create iter_decider instance using registry with proper arguments
    iter_decider_class = get_iter_decider_class(class_name)
    iter_decider = iter_decider_class(**init_args)

    # Load state dict if available - natural loading
    state_dict = data.get("state_dict", {})
    if state_dict:
        # Filter out state_dict keys that conflict with init_args
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key not in init_args:
                filtered_state_dict[key] = value
            else:
                print(f"Skipping state_dict key '{key}' as it conflicts with init_args")
        print(f"Loading iter_decider state dict with {len(filtered_state_dict)} parameters (filtered from {len(state_dict)})")
        if filtered_state_dict:
            # print(filtered_state_dict.values())
            iter_decider.load_state_dict(filtered_state_dict, strict=False)

    return iter_decider


