"""
Agent - Agent class with persistent identity and isolated private memory.

This module provides the Agent class which maintains persistent identity
through system prompts and isolated private memory to prevent context leakage
between agents.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from uuid import uuid4
import copy

from .inference_engine import InferenceEngine, InferenceRequest


@dataclass
class Message:
    """Represents a single message in dialogue."""
    role: str  # 'agent', 'user', or 'system'
    content: str
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """
    Agent with persistent identity and isolated private memory.
    
    Each agent maintains:
    - Persistent identity through system prompts
    - Isolated private memory (not shared with other agents)
    - Ability to process shared dialogue from public bus
    - Generation of responses without context leakage
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        agent_id: Optional[str] = None
    ):
        """
        Initialize an Agent.
        
        Args:
            name: Human-readable name for the agent
            system_prompt: System prompt that defines agent's identity and behavior
            temperature: Sampling temperature for inference
            max_tokens: Maximum tokens to generate
            agent_id: Optional unique identifier (generated if not provided)
        """
        self.agent_id = agent_id or str(uuid4())
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Private memory - isolated from other agents
        self._private_memory: List[Message] = []
        
        # Reference to singleton inference engine
        self._inference_engine = InferenceEngine()
        
        # Track agent state
        self._turn_count = 0
    
    def process_turn(self, shared_dialogue: List[Message]) -> str:
        """
        Process a turn with shared dialogue from the public bus.
        
        Args:
            shared_dialogue: List of messages from the shared public bus
            
        Returns:
            Generated response text
            
        Note:
            The agent can see shared_dialogue but maintains its own
            private_memory that is not exposed to other agents.
        """
        self._turn_count += 1
        
        # Build the prompt from shared dialogue
        prompt = self._build_prompt(shared_dialogue)
        
        # Create inference request
        request = InferenceRequest(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Get response from inference engine (serial processing)
        response = self._inference_engine.infer(request)
        
        # Store in private memory
        private_msg = Message(
            role="agent",
            content=response.text,
            agent_id=self.agent_id,
            metadata={
                "turn": self._turn_count,
                "inference_metadata": response.metadata
            }
        )
        self._private_memory.append(private_msg)
        
        return response.text
    
    def _build_prompt(self, shared_dialogue: List[Message]) -> str:
        """
        Build prompt from shared dialogue.
        
        Args:
            shared_dialogue: Messages from the public bus
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Include recent shared dialogue
        for msg in shared_dialogue[-10:]:  # Limit to last 10 messages
            if msg.agent_id and msg.agent_id != self.agent_id:
                # Message from another agent
                agent_name = msg.metadata.get('agent_name', 'Other')
                prompt_parts.append(f"{agent_name}: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
        
        # Optionally include relevant private memory context
        # (agent can reference its own past thoughts)
        if self._private_memory:
            last_private = self._private_memory[-1]
            prompt_parts.append(f"[My last thought: {last_private.content}]")
        
        return "\n".join(prompt_parts) if prompt_parts else "Start the conversation."
    
    def add_to_private_memory(self, message: Message):
        """
        Add a message to private memory.
        
        Args:
            message: Message to store in private memory
        """
        self._private_memory.append(copy.deepcopy(message))
    
    def get_private_memory(self) -> List[Message]:
        """
        Get a copy of private memory.
        
        Returns:
            Deep copy of private memory to prevent external modification
        """
        return copy.deepcopy(self._private_memory)
    
    def clear_private_memory(self):
        """Clear the agent's private memory."""
        self._private_memory.clear()
        self._turn_count = 0
    
    @property
    def turn_count(self) -> int:
        """Get the number of turns this agent has processed."""
        return self._turn_count
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"Agent(name='{self.name}', id='{self.agent_id[:8]}...', "
                f"turns={self._turn_count}, private_memory_size={len(self._private_memory)})")
