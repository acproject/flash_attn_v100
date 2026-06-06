from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time
import uuid


class RequestStatus(Enum):
    """Inference request lifecycle states."""
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class InferenceRequest:
    """Represents a single LLM inference request with sampling parameters and state tracking."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_text: Optional[str] = None
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    status: RequestStatus = RequestStatus.WAITING
    max_tokens: int = 128
    arrival_time: float = field(default_factory=time.time)
    completion_time: Optional[float] = None
    stop_token_ids: list[int] = field(default_factory=list)
    # Sampling params
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    # Stream callback
    stream_callback: Optional[callable] = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.COMPLETED, RequestStatus.STOPPED)
