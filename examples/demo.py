"""
Example usage of the AgentSim multi-agent framework.

This example demonstrates:
1. Singleton InferenceEngine for serial GPU processing
2. Multiple agents with persistent identities and private memory
3. Workspace orchestration with shared public bus
4. Isolated agent states without context leakage
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentsim import Agent, Workspace, InferenceEngine
from agentsim.workspace import TurnPolicy


def main():
    """Run a demonstration of the multi-agent framework."""
    
    print("="*70)
    print("AgentSim - Multi-Agent Framework Demo")
    print("="*70)
    
    # 1. Create agents with different identities
    print("\n[1] Creating agents with persistent identities...")
    
    analyst = Agent(
        name="DataAnalyst",
        system_prompt="You are a data analyst. Provide analytical insights and data-driven perspectives.",
        temperature=0.5,
        max_tokens=256
    )
    
    creative = Agent(
        name="CreativeWriter",
        system_prompt="You are a creative writer. Provide imaginative and engaging narratives.",
        temperature=0.9,
        max_tokens=256
    )
    
    critic = Agent(
        name="Critic",
        system_prompt="You are a constructive critic. Evaluate ideas and provide balanced feedback.",
        temperature=0.7,
        max_tokens=256
    )
    
    print(f"   - Created {analyst}")
    print(f"   - Created {creative}")
    print(f"   - Created {critic}")
    
    # 2. Verify singleton InferenceEngine
    print("\n[2] Verifying singleton InferenceEngine...")
    engine1 = InferenceEngine()
    engine2 = InferenceEngine()
    
    print(f"   - Engine 1 ID: {id(engine1)}")
    print(f"   - Engine 2 ID: {id(engine2)}")
    print(f"   - Same instance: {engine1 is engine2}")
    print(f"   - Device: {engine1.device}")
    
    # 3. Create workspace for orchestration
    print("\n[3] Creating workspace with turn-based orchestration...")
    
    workspace = Workspace(
        agents=[analyst, creative, critic],
        turn_policy=TurnPolicy.ROUND_ROBIN,
        max_turns=6  # 2 turns per agent
    )
    
    print(f"   - {workspace}")
    
    # 4. Run simulation
    print("\n[4] Running simulation with shared public bus...")
    initial_prompt = "Let's discuss the future of artificial intelligence."
    
    dialogue = workspace.run(initial_message=initial_prompt)
    
    # 5. Display results
    print("\n[5] Simulation complete!")
    workspace.print_dialogue()
    
    # 6. Demonstrate private memory isolation
    print("\n[6] Demonstrating private memory isolation...")
    print(f"\n   Analyst private memory (size={len(analyst.get_private_memory())}):")
    for i, msg in enumerate(analyst.get_private_memory()[:2], 1):
        print(f"      [{i}] {msg.content[:60]}...")
    
    print(f"\n   Creative private memory (size={len(creative.get_private_memory())}):")
    for i, msg in enumerate(creative.get_private_memory()[:2], 1):
        print(f"      [{i}] {msg.content[:60]}...")
    
    print(f"\n   Critic private memory (size={len(critic.get_private_memory())}):")
    for i, msg in enumerate(critic.get_private_memory()[:2], 1):
        print(f"      [{i}] {msg.content[:60]}...")
    
    # 7. Show execution history
    print("\n[7] Execution history:")
    history = workspace.get_execution_history()
    for record in history:
        print(f"   Turn {record['turn']}: {record['agent_name']} "
              f"(dialogue length: {record['dialogue_length']})")
    
    # 8. Verify no context leakage
    print("\n[8] Verifying context isolation...")
    print(f"   - Public dialogue messages: {len(dialogue)}")
    print(f"   - Analyst private memory: {len(analyst.get_private_memory())} (isolated)")
    print(f"   - Creative private memory: {len(creative.get_private_memory())} (isolated)")
    print(f"   - Critic private memory: {len(critic.get_private_memory())} (isolated)")
    print(f"   - Total inference requests: {engine1.request_count}")
    
    print("\n[9] Key features demonstrated:")
    print("   ✓ Singleton InferenceEngine for serial processing")
    print("   ✓ Agents with persistent identities (system prompts)")
    print("   ✓ Isolated private memory per agent")
    print("   ✓ Shared public bus for communication")
    print("   ✓ Turn-based orchestration via Workspace")
    print("   ✓ No context leakage between agents")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
