"""Tests for Agent class."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from agentsim import Agent, InferenceEngine
from agentsim.agent import Message


class TestAgent(unittest.TestCase):
    """Test cases for Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        InferenceEngine.reset_instance()
        self.agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent."
        )
    
    def test_agent_creation(self):
        """Test agent creation with proper initialization."""
        agent = Agent(
            name="MyAgent",
            system_prompt="Test prompt",
            temperature=0.5,
            max_tokens=256
        )
        
        self.assertEqual(agent.name, "MyAgent")
        self.assertEqual(agent.system_prompt, "Test prompt")
        self.assertEqual(agent.temperature, 0.5)
        self.assertEqual(agent.max_tokens, 256)
        self.assertIsNotNone(agent.agent_id)
        self.assertEqual(agent.turn_count, 0)
    
    def test_agent_id_uniqueness(self):
        """Test that different agents have unique IDs."""
        agent1 = Agent(name="Agent1", system_prompt="Prompt1")
        agent2 = Agent(name="Agent2", system_prompt="Prompt2")
        
        self.assertNotEqual(agent1.agent_id, agent2.agent_id)
    
    def test_custom_agent_id(self):
        """Test agent creation with custom ID."""
        custom_id = "custom-123"
        agent = Agent(
            name="CustomAgent",
            system_prompt="Test",
            agent_id=custom_id
        )
        
        self.assertEqual(agent.agent_id, custom_id)
    
    def test_process_turn(self):
        """Test agent processing a turn."""
        shared_dialogue = [
            Message(role="user", content="Hello!", metadata={"turn": 1})
        ]
        
        response = self.agent.process_turn(shared_dialogue)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertEqual(self.agent.turn_count, 1)
        self.assertEqual(len(self.agent.get_private_memory()), 1)
    
    def test_private_memory_isolation(self):
        """Test that private memory is isolated."""
        agent1 = Agent(name="Agent1", system_prompt="Prompt1")
        agent2 = Agent(name="Agent2", system_prompt="Prompt2")
        
        shared = [Message(role="user", content="Test", metadata={})]
        
        agent1.process_turn(shared)
        agent2.process_turn(shared)
        
        mem1 = agent1.get_private_memory()
        mem2 = agent2.get_private_memory()
        
        self.assertEqual(len(mem1), 1)
        self.assertEqual(len(mem2), 1)
        self.assertNotEqual(mem1[0].agent_id, mem2[0].agent_id)
    
    def test_private_memory_copy(self):
        """Test that get_private_memory returns a copy."""
        shared = [Message(role="user", content="Test", metadata={})]
        self.agent.process_turn(shared)
        
        memory_copy = self.agent.get_private_memory()
        memory_copy.append(Message(role="test", content="Fake", metadata={}))
        
        # Original should be unchanged
        self.assertEqual(len(self.agent.get_private_memory()), 1)
    
    def test_add_to_private_memory(self):
        """Test adding messages to private memory."""
        msg = Message(role="system", content="Test message", metadata={})
        self.agent.add_to_private_memory(msg)
        
        memory = self.agent.get_private_memory()
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0].content, "Test message")
    
    def test_clear_private_memory(self):
        """Test clearing private memory."""
        shared = [Message(role="user", content="Test", metadata={})]
        self.agent.process_turn(shared)
        self.agent.process_turn(shared)
        
        self.assertEqual(len(self.agent.get_private_memory()), 2)
        self.assertEqual(self.agent.turn_count, 2)
        
        self.agent.clear_private_memory()
        
        self.assertEqual(len(self.agent.get_private_memory()), 0)
        self.assertEqual(self.agent.turn_count, 0)
    
    def test_multiple_turns(self):
        """Test agent processing multiple turns."""
        shared = [Message(role="user", content="Turn 1", metadata={})]
        
        for i in range(5):
            response = self.agent.process_turn(shared)
            self.assertIsNotNone(response)
        
        self.assertEqual(self.agent.turn_count, 5)
        self.assertEqual(len(self.agent.get_private_memory()), 5)
    
    def test_agent_repr(self):
        """Test agent string representation."""
        repr_str = repr(self.agent)
        
        self.assertIn("TestAgent", repr_str)
        self.assertIn("turns=0", repr_str)
        self.assertIn("private_memory_size=0", repr_str)


if __name__ == '__main__':
    unittest.main()
