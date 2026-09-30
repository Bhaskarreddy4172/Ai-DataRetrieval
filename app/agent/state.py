"""Agent state machine enum and request lifecycle context."""

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentStateEnum(str, Enum):
    RECEIVED = "RECEIVED"
    UNDERSTANDING = "UNDERSTANDING"
    CONTEXT_RESOLUTION = "CONTEXT_RESOLUTION"
    ROUTING = "ROUTING"
    PLANNING = "PLANNING"
    TOOL_SELECTION = "TOOL_SELECTION"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    REPLANNING = "REPLANNING"
    VERIFICATION = "VERIFICATION"
    ANSWER_GENERATION = "ANSWER_GENERATION"
    COMPLETED = "COMPLETED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"


class AgentTraceEvent(BaseModel):
    state: AgentStateEnum
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class RequestContext(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(default="default")
    dataset_id: str = Field(default="default_dataset")
    user_message: str
    normalized_message: Optional[str] = None
    resolved_message: Optional[str] = None
    current_state: AgentStateEnum = AgentStateEnum.RECEIVED
    trace_events: List[AgentTraceEvent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def transition_to(self, new_state: AgentStateEnum, message: str, details: Optional[Dict[str, Any]] = None):
        """Record state machine transition event."""
        self.current_state = new_state
        event = AgentTraceEvent(
            state=new_state,
            message=message,
            details=details or {},
            timestamp=round(time.time(), 3),
        )
        self.trace_events.append(event)

