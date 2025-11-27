import functools
import logging
from dataclasses import replace
from typing import Any
from pocket_joe.core import Action, Context, Policy

logger = logging.getLogger(__name__)

def loop_wrapper(max_turns: int = 10):
    """
    Decorator: Repeats the wrapped policy until a 'done' signal or max_turns.
    Manages state transitions when the inner policy returns a new Action.
    """
    def decorator(func: Policy):
        @functools.wraps(func)
        async def wrapper(action: Action, ctx: Context) -> Any:
            logger.debug(f"Starting Loop (Max {max_turns})")
            
            current_action = action
            
            for i in range(max_turns):
                logger.debug(f"Turn {i+1}")
                
                # 1. Run the policy logic (The "Decider")
                result = await func(current_action, ctx)
                
                # 2. Check for State Update (Continuation)
                if isinstance(result, Action):
                    logger.debug("State Updated (Continuing Loop)")
                    current_action = result
                    continue
                
                # 3. Check for Termination
                if isinstance(result, dict) and result.get("done", False):
                    logger.debug("Loop Finished (Done)")
                    return result.get("value")
                
                # 4. Fallback (e.g. if inner policy returns something unexpected)
                logger.warning(f"Unexpected result type: {type(result)}. Aborting.")
                return result
            
            logger.warning("Loop Finished (Max Turns)")
            return "Timeout"
        return wrapper
    return decorator

def invoke_action():
    """
    Decorator: Inspects the Policy's decision and executes the requested tool.
    Returns a NEW Action with updated history if a tool was called.
    """
    def decorator(func: Policy):
        @functools.wraps(func)
        async def wrapper(action: Action, ctx: Context) -> Any:
            
            # 1. Run the policy to get the decision
            decision = await func(action, ctx)
            
            # 2. If the policy wants to call a tool...
            if isinstance(decision, dict) and "tool_call" in decision:
                tool_name = decision["tool_call"]
                tool_args = decision["tool_args"]
                
                # Security Check: Is this tool in the allowed 'edges'?
                if tool_name not in action.edges:
                    raise ValueError(f"Security Violation: Policy tried to call {tool_name}, but it was not in allowed edges: {action.edges}")

                logger.info(f"Invoking Tool: {tool_name}")
                
                # 3. Execute the tool (Durable Call)
                # Note: We don't pass decorators here, tools are usually atomic
                tool_result = await ctx.call(tool_name, Action(payload=tool_args))
                
                # 4. Create NEW History (Immutable append)
                new_history_item = {
                    "role": "tool", 
                    "name": tool_name, 
                    "content": tool_result
                }
                new_history = action.history + (new_history_item,)
                
                # 5. Return NEW Action (State Transition)
                return replace(action, history=new_history)
            
            # If no tool call, just return the decision (maybe it's 'done')
            return decision
            
        return wrapper
    return decorator
