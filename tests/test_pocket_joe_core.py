import unittest
import asyncio
from dataclasses import replace
from pocket_joe.core import Action, Context
from pocket_joe.registry import Registry
from pocket_joe.memory_runtime import InMemoryRunner
from pocket_joe.policy_decorators import loop_wrapper, invoke_action

class TestPocketJoeCore(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Reset registry for each test
        self.registry = Registry()
        self.runner = InMemoryRunner(registry=self.registry)

    async def test_action_immutability(self):
        """Test that Action is immutable and replace works as expected."""
        a1 = Action(payload="initial")
        
        # Verify immutability
        with self.assertRaises(Exception):
            a1.payload = "changed"
            
        # Verify replace
        a2 = replace(a1, payload="updated")
        self.assertEqual(a1.payload, "initial")
        self.assertEqual(a2.payload, "updated")
        # replace() keeps the original ID unless specified, which is correct for updating state of same logical action
        self.assertEqual(a1.id, a2.id)

    async def test_simple_policy_execution(self):
        """Test registering and running a simple policy."""
        
        @self.registry.register("echo")
        async def echo_policy(action: Action, ctx: Context):
            return f"Echo: {action.payload}"
            
        result = await self.runner.execute("echo", Action(payload="Hello"))
        self.assertEqual(result, "Echo: Hello")

    async def test_nested_policy_call(self):
        """Test a policy calling another policy via Context."""
        
        @self.registry.register("worker")
        async def worker(action: Action, ctx: Context):
            return action.payload * 2
            
        @self.registry.register("manager")
        async def manager(action: Action, ctx: Context):
            # Call worker
            res = await ctx.call("worker", Action(payload=10))
            return f"Worker said: {res}"
            
        result = await self.runner.execute("manager", Action(payload="start"))
        self.assertEqual(result, "Worker said: 20")

    async def test_decorators_loop_and_invoke(self):
        """Test the loop_wrapper and invoke_action decorators."""
        
        # 1. Define a tool
        @self.registry.register("tool")
        async def tool(action: Action, ctx: Context):
            return "ToolResult"
            
        # 2. Define a decider that uses the tool then finishes
        @self.registry.register("decider")
        async def decider(action: Action, ctx: Context):
            if not action.history:
                return {"tool_call": "tool", "tool_args": "arg"}
            else:
                # Check history
                last_msg = action.history[-1]
                if last_msg["role"] == "tool" and last_msg["content"] == "ToolResult":
                    return {"done": True, "value": "Success"}
                return {"done": True, "value": "Fail"}

        # 3. Define the agent that wires them up
        @self.registry.register("agent")
        async def agent(action: Action, ctx: Context):
            # Allow the tool
            scoped_action = replace(action, edges=("tool",))
            
            return await ctx.call(
                "decider", 
                scoped_action, 
                decorators=[loop_wrapper(max_turns=5), invoke_action()]
            )

        result = await self.runner.execute("agent", Action(payload="start"))
        self.assertEqual(result, "Success")

if __name__ == "__main__":
    unittest.main()
