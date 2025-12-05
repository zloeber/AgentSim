# GitHub Copilot Instructions for AgentSim

## Project Overview

AgentSim is a Python multi-agent framework designed for a single local GPU with serial inference processing. The framework enables multiple AI agents to interact in a controlled environment while maintaining:

- **Singleton InferenceEngine**: Serial GPU processing to prevent resource conflicts
- **Agent Identity**: Persistent identity through system prompts
- **Private Memory**: Isolated agent memory to prevent context leakage
- **Shared Communication**: Public bus for agent-to-agent communication
- **Turn-based Orchestration**: Workspace manages agent interactions

## Repository Structure

```
AgentSim/
├── agentsim/              # Main package
│   ├── __init__.py       # Package exports
│   ├── inference_engine.py  # Singleton GPU inference handler
│   ├── agent.py          # Agent with private memory
│   └── workspace.py      # Turn-based orchestrator
├── examples/
│   └── demo.py           # Example usage
├── tests/
│   ├── test_inference_engine.py
│   ├── test_agent.py
│   └── test_workspace.py
├── README.md
├── pyproject.toml        # Project configuration
└── requirements.txt      # Dependencies
```

## Development Setup

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ --cov=agentsim --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_agent.py -v

# Run with verbose output
python -m pytest tests/ -v
```

### Type Checking

```bash
# Run mypy for type checking
mypy agentsim/
```

## Coding Standards and Conventions

### Python Style

- **Python Version**: Target Python 3.8+
- **Type Hints**: Use type hints for function signatures and important variables
- **Docstrings**: Use Google-style docstrings for all public classes and methods
- **Line Length**: Keep lines under 100 characters when practical
- **Imports**: Group imports as: stdlib, third-party, local (separated by blank lines)

### Naming Conventions

- **Classes**: PascalCase (e.g., `InferenceEngine`, `Agent`, `Workspace`)
- **Functions/Methods**: snake_case (e.g., `process_turn`, `add_user_message`)
- **Private Members**: Prefix with underscore (e.g., `_private_memory`, `_inference_lock`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_TOKENS`)
- **Type Aliases**: PascalCase (e.g., `InferenceRequest`, `Message`)

### Code Organization

- **Dataclasses**: Use `@dataclass` for simple data containers (e.g., `Message`, `InferenceRequest`)
- **Enums**: Use `Enum` for fixed sets of values (e.g., `TurnPolicy`)
- **Thread Safety**: Use `threading.Lock` for thread-safe operations
- **Immutability**: Return deep copies when exposing internal state (e.g., `get_private_memory()`)

## Architecture and Design Principles

### 1. Singleton Pattern (InferenceEngine)

The `InferenceEngine` uses the singleton pattern to ensure only one instance exists:

```python
# Thread-safe singleton implementation
class InferenceEngine:
    _instance: Optional['InferenceEngine'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Check if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
            self._initialized = True
            # Initialize model here...
```

**Critical**: Always use double-check locking for thread safety.

### 2. Private Memory Isolation (Agent)

Each agent maintains private memory that is NOT shared with other agents:

```python
# Private memory is isolated
self._private_memory: List[Message] = []

# Return deep copies to prevent external modification
def get_private_memory(self) -> List[Message]:
    return copy.deepcopy(self._private_memory)
```

**Critical**: Never directly expose `_private_memory` - always return deep copies.

### 3. Shared Public Bus (Workspace)

The workspace manages a shared dialogue visible to all agents:

```python
# Public dialogue shared across agents
self._public_dialogue: List[Message] = []

# Agents process shared dialogue but maintain private state
response = agent.process_turn(self._public_dialogue)
```

**Critical**: Public dialogue is shared, but agent internal state remains isolated.

### 4. Serial Processing

All inference requests are processed serially to prevent GPU conflicts:

```python
def infer(self, request: InferenceRequest) -> InferenceResponse:
    with self._inference_lock:
        # Serial processing ensures no GPU conflicts
        response = self._model.generate(...)
```

## Testing Practices

### Test Structure

- **Test Classes**: Use `TestClassName` pattern (e.g., `TestAgent`, `TestWorkspace`)
- **Test Methods**: Use `test_feature_description` pattern (e.g., `test_private_memory_isolation`)
- **Assertions**: Use descriptive assertion messages
- **Fixtures**: Keep test setup minimal and inline when possible

### Key Test Patterns

```python
# Test singleton behavior
def test_singleton_pattern(self):
    engine1 = InferenceEngine()
    engine2 = InferenceEngine()
    assert engine1 is engine2

# Test isolation
def test_private_memory_isolation(self):
    agent1_memory = agent1.get_private_memory()
    agent2_memory = agent2.get_private_memory()
    assert len(agent1_memory) == 0
    assert len(agent2_memory) == 0

# Test thread safety
def test_thread_safety(self):
    def worker():
        engine.infer(request)
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```

### Test Coverage Expectations

- Aim for >90% code coverage
- All public APIs must have tests
- Test both success and error cases
- Test thread safety for concurrent operations

## Common Patterns and Idioms

### 1. Creating Agents

```python
agent = Agent(
    name="AgentName",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=512
)
```

### 2. Setting Up Workspace

```python
workspace = Workspace(
    agents=[agent1, agent2],
    turn_policy=TurnPolicy.ROUND_ROBIN,
    max_turns=10
)
```

### 3. Running Simulation

```python
# Add initial message and run
dialogue = workspace.run(initial_message="Hello!")

# Or step through manually
workspace.add_user_message("Hello!")
workspace.step()  # One agent responds
workspace.step()  # Next agent responds
```

### 4. Accessing Results

```python
# Get public dialogue (shared)
public_dialogue = workspace.get_public_dialogue()

# Get agent private memory (isolated)
private_memory = agent.get_private_memory()

# Print formatted dialogue
workspace.print_dialogue()
```

## Common Pitfalls and How to Avoid Them

### ❌ Don't expose mutable internal state

```python
# BAD - exposes internal list
def get_private_memory(self):
    return self._private_memory
```

```python
# GOOD - returns deep copy
def get_private_memory(self):
    return copy.deepcopy(self._private_memory)
```

### ❌ Don't break singleton pattern

```python
# BAD - creates multiple instances
class InferenceEngine:
    def __init__(self):
        self.model = load_model()
```

```python
# GOOD - maintains thread-safe singleton with type hints
class InferenceEngine:
    _instance: Optional['InferenceEngine'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
```

### ❌ Don't share memory between agents

```python
# BAD - shared memory leaks context
class Workspace:
    def __init__(self):
        self.shared_memory = []
        for agent in agents:
            agent.memory = self.shared_memory
```

```python
# GOOD - isolated private memory
class Agent:
    def __init__(self):
        self._private_memory = []  # Isolated per agent
```

### ❌ Don't bypass inference serialization

```python
# BAD - concurrent GPU access
def process_turn(self):
    thread = Thread(target=self._inference_engine.infer)
    thread.start()
```

```python
# GOOD - serial processing through singleton
def process_turn(self):
    response = self._inference_engine.infer(request)
```

## Mock Implementation Notes

The current implementation uses **mock inference** for demonstration purposes:

- `InferenceEngine._mock_inference()` returns simple echo responses
- No actual GPU model is loaded
- Easy to swap in real models (transformers, vLLM, etc.)

### Future Integration Points

To integrate real GPU models:

1. Update `InferenceEngine._initialize_model()` to load actual model
2. Replace `_mock_inference()` with real inference call
3. Add model-specific dependencies to `requirements.txt`
4. Ensure serial processing through `_inference_lock` is maintained

Example:

```python
def _initialize_model(self):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    self.tokenizer = AutoTokenizer.from_pretrained("model-name")
    self.model = AutoModelForCausalLM.from_pretrained("model-name")
    self.model.to("cuda:0")
```

## Additional Guidelines

### When Adding New Features

1. **Maintain isolation**: Ensure agent private memory remains isolated
2. **Preserve serialization**: Keep inference processing serial
3. **Add tests**: Write tests before implementation when possible
4. **Update docstrings**: Document all public APIs
5. **Check thread safety**: Use locks for shared state

### When Modifying Core Classes

- `InferenceEngine`: Maintain singleton pattern and serial processing
- `Agent`: Preserve private memory isolation and identity
- `Workspace`: Keep public bus shared and agent state isolated

### Documentation Standards

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Keep this file updated with new patterns

## Questions or Issues?

If you're unsure about:
- Design decisions: Check this file and architecture section
- Code patterns: Look at existing code in the same module
- Testing approach: Check similar tests in `tests/` directory
- Thread safety: Review `InferenceEngine` singleton implementation

For major changes, consider discussing the approach before implementation.
