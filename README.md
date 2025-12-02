# AgentSim

A Python multi-agent framework designed for a single local GPU with serial inference processing.

## Overview

AgentSim provides a clean, efficient framework for building multi-agent systems that:
- **Singleton InferenceEngine**: Ensures serial processing on a single GPU to prevent resource conflicts
- **Agent Identity**: Each agent maintains persistent identity through system prompts
- **Private Memory**: Agents have isolated private memory to prevent context leakage
- **Shared Communication**: Agents communicate via a shared public bus
- **Workspace Orchestration**: Turn-based orchestrator manages agent interactions

## Architecture

### Core Components

1. **InferenceEngine** (Singleton)
   - Handles all GPU inference requests serially
   - Thread-safe processing queue
   - Prevents resource conflicts on single GPU

2. **Agent**
   - Persistent identity via system prompts
   - Isolated private memory (not shared between agents)
   - Processes shared dialogue from public bus
   - Maintains distinct internal state

3. **Workspace**
   - Orchestrates turn-based execution
   - Manages shared public dialogue bus
   - Configurable turn policies (round-robin, random, custom)
   - Ensures no context leakage between agents

## Installation

```bash
# Clone the repository
git clone https://github.com/zloeber/AgentSim.git
cd AgentSim

# Install dependencies (optional, for testing)
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from agentsim import Agent, Workspace, InferenceEngine
from agentsim.workspace import TurnPolicy

# Create agents with different identities
analyst = Agent(
    name="DataAnalyst",
    system_prompt="You are a data analyst. Provide analytical insights.",
    temperature=0.5
)

creative = Agent(
    name="CreativeWriter",
    system_prompt="You are a creative writer. Provide imaginative narratives.",
    temperature=0.9
)

# Create workspace for orchestration
workspace = Workspace(
    agents=[analyst, creative],
    turn_policy=TurnPolicy.ROUND_ROBIN,
    max_turns=6
)

# Run simulation
dialogue = workspace.run(initial_message="Let's discuss AI!")

# Display results
workspace.print_dialogue()
```

## Key Features

### ✓ Singleton InferenceEngine
Ensures all inference happens serially on a single GPU:

```python
engine1 = InferenceEngine()
engine2 = InferenceEngine()
assert engine1 is engine2  # Same instance
```

### ✓ Isolated Private Memory
Each agent maintains private memory that other agents cannot access:

```python
# Agent 1 has its own private memory
agent1_memory = agent1.get_private_memory()

# Agent 2 has completely separate memory
agent2_memory = agent2.get_private_memory()

# No context leakage between agents
```

### ✓ Shared Public Bus
Agents communicate via shared dialogue visible to all:

```python
# All agents can see public dialogue
workspace.add_user_message("Hello everyone!")
workspace.run()

# But private memories remain isolated
```

### ✓ Turn-Based Orchestration
Workspace manages agent turns with configurable policies:

```python
# Round-robin turns
workspace = Workspace(agents=[a1, a2], turn_policy=TurnPolicy.ROUND_ROBIN)

# Random selection
workspace = Workspace(agents=[a1, a2], turn_policy=TurnPolicy.RANDOM)

# Custom policy
def custom_selector(agents, dialogue):
    return agents[0]  # Your logic here

workspace = Workspace(
    agents=[a1, a2],
    turn_policy=TurnPolicy.CUSTOM,
    custom_turn_selector=custom_selector
)
```

## Examples

Run the demo:

```bash
python examples/demo.py
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=agentsim --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_agent.py -v
```

## Project Structure

```
AgentSim/
├── agentsim/              # Main package
│   ├── __init__.py
│   ├── inference_engine.py  # Singleton GPU inference
│   ├── agent.py             # Agent with private memory
│   └── workspace.py         # Orchestrator
├── examples/
│   └── demo.py            # Example usage
├── tests/
│   ├── test_inference_engine.py
│   ├── test_agent.py
│   └── test_workspace.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Design Principles

1. **Single GPU Optimization**: All inference is serialized through a singleton to prevent GPU memory conflicts
2. **Context Isolation**: Private memories ensure agents maintain distinct internal states
3. **Clean Communication**: Shared public bus for transparent agent-to-agent communication
4. **Extensibility**: Easy to integrate real GPU models (transformers, vLLM, etc.)
5. **Thread Safety**: Built-in thread-safe operations for concurrent agent management

## Future Extensions

The current implementation uses mock inference. To integrate real GPU models:

```python
# In inference_engine.py _initialize_model():
from transformers import AutoModelForCausalLM, AutoTokenizer

self.tokenizer = AutoTokenizer.from_pretrained("model-name")
self.model = AutoModelForCausalLM.from_pretrained("model-name")
self.model.to("cuda:0")

# In _mock_inference(), replace with:
inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
outputs = self.model.generate(**inputs, max_new_tokens=request.max_tokens)
return self.tokenizer.decode(outputs[0])
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details