"""Tests for Workspace orchestrator."""

import unittest
from agentsim import Agent, Workspace, InferenceEngine
from agentsim.workspace import TurnPolicy
from agentsim.agent import Message


class TestWorkspace(unittest.TestCase):
    """Test cases for Workspace."""
    
    def setUp(self):
        """Set up test fixtures."""
        InferenceEngine.reset_instance()
        
        self.agent1 = Agent(name="Agent1", system_prompt="You are agent 1.")
        self.agent2 = Agent(name="Agent2", system_prompt="You are agent 2.")
        self.agents = [self.agent1, self.agent2]
    
    def test_workspace_creation(self):
        """Test workspace creation."""
        workspace = Workspace(
            agents=self.agents,
            turn_policy=TurnPolicy.ROUND_ROBIN,
            max_turns=5
        )
        
        self.assertEqual(len(workspace.agents), 2)
        self.assertEqual(workspace.max_turns, 5)
        self.assertEqual(workspace.turn_policy, TurnPolicy.ROUND_ROBIN)
    
    def test_workspace_requires_agents(self):
        """Test that workspace requires at least one agent."""
        with self.assertRaises(ValueError):
            Workspace(agents=[], max_turns=5)
    
    def test_add_user_message(self):
        """Test adding user messages to public bus."""
        workspace = Workspace(agents=self.agents, max_turns=5)
        
        workspace.add_user_message("Hello!")
        dialogue = workspace.get_public_dialogue()
        
        self.assertEqual(len(dialogue), 1)
        self.assertEqual(dialogue[0].role, "user")
        self.assertEqual(dialogue[0].content, "Hello!")
    
    def test_round_robin_turn_policy(self):
        """Test round-robin turn policy."""
        workspace = Workspace(
            agents=self.agents,
            turn_policy=TurnPolicy.ROUND_ROBIN,
            max_turns=4
        )
        
        workspace.run(initial_message="Start")
        history = workspace.get_execution_history()
        
        # Check that agents alternate
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]['agent_name'], "Agent1")
        self.assertEqual(history[1]['agent_name'], "Agent2")
        self.assertEqual(history[2]['agent_name'], "Agent1")
        self.assertEqual(history[3]['agent_name'], "Agent2")
    
    def test_workspace_run(self):
        """Test workspace execution."""
        workspace = Workspace(agents=self.agents, max_turns=3)
        
        dialogue = workspace.run(initial_message="Hello agents!")
        
        # Initial message + 3 agent responses
        self.assertEqual(len(dialogue), 4)
        self.assertEqual(dialogue[0].role, "user")
        self.assertEqual(dialogue[1].role, "agent")
        self.assertEqual(dialogue[2].role, "agent")
        self.assertEqual(dialogue[3].role, "agent")
    
    def test_workspace_step(self):
        """Test single step execution."""
        workspace = Workspace(agents=self.agents, max_turns=5)
        workspace.add_user_message("Test")
        
        msg1 = workspace.step()
        self.assertIsNotNone(msg1)
        self.assertEqual(msg1.role, "agent")
        
        msg2 = workspace.step()
        self.assertIsNotNone(msg2)
        self.assertEqual(msg2.role, "agent")
        
        # Check dialogue length
        dialogue = workspace.get_public_dialogue()
        self.assertEqual(len(dialogue), 3)  # 1 user + 2 agent messages
    
    def test_max_turns_limit(self):
        """Test that workspace respects max_turns limit."""
        workspace = Workspace(agents=self.agents, max_turns=2)
        
        dialogue = workspace.run(initial_message="Start")
        
        # Initial message + max_turns agent responses
        self.assertEqual(len(dialogue), 3)
    
    def test_public_dialogue_shared(self):
        """Test that public dialogue is shared across agents."""
        workspace = Workspace(agents=self.agents, max_turns=2)
        workspace.run(initial_message="Shared message")
        
        # Both agents should have seen the shared dialogue
        self.assertEqual(self.agent1.turn_count, 1)
        self.assertEqual(self.agent2.turn_count, 1)
        
        # Public dialogue should have all messages
        dialogue = workspace.get_public_dialogue()
        self.assertGreater(len(dialogue), 0)
    
    def test_private_memory_isolation(self):
        """Test that private memories remain isolated."""
        workspace = Workspace(agents=self.agents, max_turns=2)
        workspace.run(initial_message="Test isolation")
        
        mem1 = self.agent1.get_private_memory()
        mem2 = self.agent2.get_private_memory()
        
        # Each agent should have its own private memory
        self.assertEqual(len(mem1), 1)
        self.assertEqual(len(mem2), 1)
        
        # Memories should be different
        self.assertNotEqual(mem1[0].agent_id, mem2[0].agent_id)
    
    def test_execution_history(self):
        """Test execution history tracking."""
        workspace = Workspace(agents=self.agents, max_turns=3)
        workspace.run(initial_message="Test")
        
        history = workspace.get_execution_history()
        
        self.assertEqual(len(history), 3)
        for record in history:
            self.assertIn('turn', record)
            self.assertIn('agent_id', record)
            self.assertIn('agent_name', record)
            self.assertIn('response', record)
    
    def test_workspace_reset(self):
        """Test workspace reset functionality."""
        workspace = Workspace(agents=self.agents, max_turns=3)
        workspace.run(initial_message="Test")
        
        self.assertGreater(len(workspace.get_public_dialogue()), 0)
        
        workspace.reset()
        
        self.assertEqual(len(workspace.get_public_dialogue()), 0)
        self.assertEqual(len(workspace.get_execution_history()), 0)
    
    def test_stop_conditions(self):
        """Test early stopping conditions."""
        workspace = Workspace(agents=self.agents, max_turns=10)
        
        # Mock a response that triggers stop
        # Note: This is a simplified test since our mock doesn't actually
        # generate [DONE] responses
        workspace.run(initial_message="Test")
        
        # Just verify it completes without error
        self.assertGreater(len(workspace.get_public_dialogue()), 0)
    
    def test_custom_turn_policy(self):
        """Test custom turn policy."""
        def custom_selector(agents, dialogue):
            # Always select first agent
            return agents[0]
        
        workspace = Workspace(
            agents=self.agents,
            turn_policy=TurnPolicy.CUSTOM,
            max_turns=3,
            custom_turn_selector=custom_selector
        )
        
        workspace.run(initial_message="Test")
        history = workspace.get_execution_history()
        
        # All turns should be from Agent1
        for record in history:
            self.assertEqual(record['agent_name'], "Agent1")
    
    def test_custom_turn_policy_requires_selector(self):
        """Test that CUSTOM policy requires a selector function."""
        with self.assertRaises(ValueError):
            Workspace(
                agents=self.agents,
                turn_policy=TurnPolicy.CUSTOM,
                max_turns=5
            )
    
    def test_workspace_repr(self):
        """Test workspace string representation."""
        workspace = Workspace(agents=self.agents, max_turns=5)
        repr_str = repr(workspace)
        
        self.assertIn("agents=2", repr_str)
        self.assertIn("turns=0/5", repr_str)


if __name__ == '__main__':
    unittest.main()
