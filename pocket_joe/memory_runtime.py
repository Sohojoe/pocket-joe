from collections.abc import Callable

from .core import BaseContext

class InMemoryRunner:
    def _bind_strategy(self, policy: Callable, ctx: BaseContext):
        """Bind a policy function to the context.
        
        Args:
            policy: The policy function to bind
            ctx: The context instance
            
        Returns:
            Async function that wraps the policy with options execution
        """
        from .policy_wrappers import invoke_options_wrapper_for_func
        
        async def bound(**kwargs):
            wrapped = invoke_options_wrapper_for_func(policy, ctx)
            selected_actions = await wrapped(**kwargs)
            return selected_actions

        return bound

