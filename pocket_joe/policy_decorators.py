import asyncio
from typing import Callable, Any
from pocket_joe.core import Message, Policy


def invoke_action_wrapper() -> Callable[[Policy], Policy]:
    """
    Decorator that executes any options (action_call messages) in parallel after the wrapped policy completes.
    
    After the wrapped policy returns selected_actions, this decorator:
    1. Enumerates through each returned message
    2. Finds any messages with type="action_call" 
    3. Executes those actions in parallel via ctx.call()
    4. Waits for all options executions to complete
    
    Exceptions from options will propagate up the stack.
    """
    def decorator(policy_func: Policy) -> Policy:
        async def wrapper(action: Action, ctx: Context, **kwargs: Any) -> list[Message]:
            # Execute the wrapped policy
            selected_actions = await policy_func(action, ctx, **kwargs)
            
            # Find all action_call messages
            action_calls = [
                msg for msg in selected_actions 
                if msg.type == "action_call" and isinstance(msg.payload, dict)
            ]
            
            if not action_calls:
                return selected_actions
            
            # Execute all action_calls in parallel
            async def execute_option(option: Message) -> list[Message]:
                """Execute a single action_call option."""
                payload_dict = option.payload
                policy_name = payload_dict.get("policy")
                
                if not policy_name:
                    return []
                
                # Build Action from the step
                substep_action = Action(
                    policy=policy_name,
                    # TODO: Should we include text responses as well?
                    payload=action.payload + [option],  # Include history + this action_call
                    actions=action.actions
                )
                
                # Execute the substep
                return await ctx.call(substep_action)
            
            # Execute all substeps in parallel and wait for completion
            # Exceptions will propagate up the stack
            option_selected_actions = await asyncio.gather(
                *[execute_option(option) for option in action_calls]
            )
            
            # Flatten results
            all_option_selected_actions = []
            for result in option_selected_actions:
                all_option_selected_actions.extend(result)
            
            return selected_actions + all_option_selected_actions
        
        return wrapper
    return decorator
