import asyncio
from typing import Callable, Any
from pocket_joe.core import Action, Context, Step, Policy


def invoke_action_wrapper() -> Callable[[Policy], Policy]:
    """
    Decorator that executes any substeps (action_call steps) in parallel after the wrapped policy completes.
    
    After the wrapped policy returns steps, this decorator:
    1. Enumerates through each returned step
    2. Finds any steps with type="action_call" 
    3. Executes those actions in parallel via ctx.call()
    4. Waits for all substep executions to complete
    
    Exceptions from substeps will propagate up the stack.
    """
    def decorator(policy_func: Policy) -> Policy:
        async def wrapper(action: Action, ctx: Context, **kwargs: Any) -> list[Step]:
            # Execute the wrapped policy
            steps = await policy_func(action, ctx, **kwargs)
            
            # Find all action_call steps
            action_calls = [
                step for step in steps 
                if step.type == "action_call" and isinstance(step.payload, dict)
            ]
            
            if not action_calls:
                return steps
            
            # Execute all action_calls in parallel
            async def execute_substep(step: Step) -> list[Step]:
                """Execute a single action_call step."""
                payload_dict = step.payload
                policy_name = payload_dict.get("policy")
                
                if not policy_name:
                    return []
                
                # Build Action from the step
                substep_action = Action(
                    policy=policy_name,
                    # TODO: Should we include text responses as well?
                    payload=action.payload + [step],  # Include history + this action_call
                    actions=action.actions
                )
                
                # Execute the substep
                return await ctx.call(substep_action)
            
            # Execute all substeps in parallel and wait for completion
            # Exceptions will propagate up the stack
            substep_results = await asyncio.gather(
                *[execute_substep(step) for step in action_calls]
            )
            
            # Flatten results
            all_substeps = []
            for result in substep_results:
                all_substeps.extend(result)
            
            return steps + all_substeps
        
        return wrapper
    return decorator
