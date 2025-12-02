"""
Workspace - Orchestrator for managing turn-based multi-agent interactions.

This module provides the Workspace class which orchestrates turn-based loops,
manages the shared public bus for agent communication, and ensures agents
maintain distinct internal states without context leakage.
"""

from typing import List, Optional, Dict, Any, Callable
from enum import Enum

from .agent import Agent, Message


class TurnPolicy(Enum):
    """Policies for determining turn order."""
    ROUND_ROBIN = "round_robin"  # Agents take turns in order
    RANDOM = "random"  # Random agent selection
    CUSTOM = "custom"  # Custom selection function


class Workspace:
    """
    Workspace orchestrator for managing multi-agent interactions.
    
    The Workspace:
    - Manages a shared public bus (dialogue) visible to all agents
    - Orchestrates turn-based execution
    - Ensures isolation of agent private memory
    - Prevents context leakage between agents
    - Provides hooks for custom turn policies
    """
    
    def __init__(
        self,
        agents: List[Agent],
        turn_policy: TurnPolicy = TurnPolicy.ROUND_ROBIN,
        max_turns: int = 10,
        custom_turn_selector: Optional[Callable[[List[Agent], List[Message]], Agent]] = None
    ):
        """
        Initialize the Workspace.
        
        Args:
            agents: List of Agent objects to participate
            turn_policy: Policy for determining turn order
            max_turns: Maximum number of turns to execute
            custom_turn_selector: Custom function to select next agent
                                 (required if turn_policy is CUSTOM)
        """
        if not agents:
            raise ValueError("Workspace requires at least one agent")
        
        self.agents = agents
        self.turn_policy = turn_policy
        self.max_turns = max_turns
        self.custom_turn_selector = custom_turn_selector
        
        # Shared public bus - visible to all agents
        self._public_dialogue: List[Message] = []
        
        # Track workspace state
        self._current_turn = 0
        self._current_agent_idx = 0
        self._execution_history: List[Dict[str, Any]] = []
        
        # Validate custom selector if needed
        if turn_policy == TurnPolicy.CUSTOM and custom_turn_selector is None:
            raise ValueError("custom_turn_selector required when turn_policy is CUSTOM")
    
    def add_user_message(self, content: str):
        """
        Add a user message to the shared public bus.
        
        Args:
            content: The user message content
        """
        msg = Message(
            role="user",
            content=content,
            metadata={"turn": self._current_turn}
        )
        self._public_dialogue.append(msg)
    
    def run(self, initial_message: Optional[str] = None) -> List[Message]:
        """
        Run the workspace for the configured number of turns.
        
        Args:
            initial_message: Optional initial user message to start conversation
            
        Returns:
            The complete public dialogue after all turns
        """
        # Add initial message if provided
        if initial_message:
            self.add_user_message(initial_message)
        
        # Execute turn-based loop
        for turn in range(self.max_turns):
            self._current_turn = turn + 1
            
            # Select next agent based on policy
            agent = self._select_next_agent()
            
            # Agent processes turn with shared dialogue
            response = agent.process_turn(self._public_dialogue)
            
            # Parse and add response to public bus
            public_msg = self._parse_agent_output(agent, response)
            self._public_dialogue.append(public_msg)
            
            # Record execution history
            self._execution_history.append({
                "turn": self._current_turn,
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "response": response,
                "dialogue_length": len(self._public_dialogue)
            })
            
            # Check for stop conditions
            if self._should_stop(response):
                break
        
        return self._public_dialogue
    
    def step(self) -> Optional[Message]:
        """
        Execute a single turn (one agent response).
        
        Returns:
            The message added to public bus, or None if max turns reached
        """
        if self._current_turn >= self.max_turns:
            return None
        
        self._current_turn += 1
        
        # Select next agent
        agent = self._select_next_agent()
        
        # Agent processes turn
        response = agent.process_turn(self._public_dialogue)
        
        # Parse and add to public bus
        public_msg = self._parse_agent_output(agent, response)
        self._public_dialogue.append(public_msg)
        
        # Record history
        self._execution_history.append({
            "turn": self._current_turn,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "response": response,
            "dialogue_length": len(self._public_dialogue)
        })
        
        return public_msg
    
    def _select_next_agent(self) -> Agent:
        """
        Select the next agent based on the turn policy.
        
        Returns:
            The selected Agent
        """
        if self.turn_policy == TurnPolicy.ROUND_ROBIN:
            agent = self.agents[self._current_agent_idx]
            self._current_agent_idx = (self._current_agent_idx + 1) % len(self.agents)
            return agent
        
        elif self.turn_policy == TurnPolicy.RANDOM:
            import random
            return random.choice(self.agents)
        
        elif self.turn_policy == TurnPolicy.CUSTOM:
            return self.custom_turn_selector(self.agents, self._public_dialogue)
        
        else:
            raise ValueError(f"Unknown turn policy: {self.turn_policy}")
    
    def _parse_agent_output(self, agent: Agent, response: str) -> Message:
        """
        Parse agent output and create a message for the public bus.
        
        Args:
            agent: The agent that generated the response
            response: The raw response text
            
        Returns:
            Message object for the public bus
        """
        return Message(
            role="agent",
            content=response,
            agent_id=agent.agent_id,
            metadata={
                "agent_name": agent.name,
                "turn": self._current_turn
            }
        )
    
    def _should_stop(self, response: str) -> bool:
        """
        Determine if execution should stop early.
        
        Args:
            response: The latest agent response
            
        Returns:
            True if execution should stop
        """
        # Simple stop condition - can be customized
        stop_phrases = ["[DONE]", "[END]", "[STOP]"]
        return any(phrase in response.upper() for phrase in stop_phrases)
    
    def get_public_dialogue(self) -> List[Message]:
        """
        Get a copy of the public dialogue.
        
        Returns:
            Copy of the public dialogue to prevent external modification
        """
        import copy
        return copy.deepcopy(self._public_dialogue)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history.
        
        Returns:
            List of execution records
        """
        import copy
        return copy.deepcopy(self._execution_history)
    
    def reset(self):
        """Reset the workspace state."""
        self._public_dialogue.clear()
        self._current_turn = 0
        self._current_agent_idx = 0
        self._execution_history.clear()
        
        # Note: We don't clear agent private memories here
        # That's the responsibility of the agents themselves
    
    def print_dialogue(self):
        """Print the public dialogue in a readable format."""
        print("\n" + "="*60)
        print("PUBLIC DIALOGUE")
        print("="*60)
        
        for i, msg in enumerate(self._public_dialogue, 1):
            if msg.role == "user":
                print(f"\n[{i}] USER:")
            elif msg.role == "agent":
                agent_name = msg.metadata.get("agent_name", "Unknown")
                print(f"\n[{i}] {agent_name.upper()}:")
            else:
                print(f"\n[{i}] {msg.role.upper()}:")
            
            print(f"    {msg.content}")
        
        print("\n" + "="*60)
    
    def __repr__(self) -> str:
        """String representation of the workspace."""
        return (f"Workspace(agents={len(self.agents)}, "
                f"turns={self._current_turn}/{self.max_turns}, "
                f"dialogue_length={len(self._public_dialogue)})")
