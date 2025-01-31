"""Example of using SubroutineTemplate for solving math problems"""

from typing import Generator, Optional

from prompttrail.agent.subroutine import SubroutineTemplate
from prompttrail.agent.subroutine.session_init_strategy import InheritSystemStrategy
from prompttrail.agent.subroutine.squash_strategy import FilterByRoleStrategy
from prompttrail.agent.templates import Stack, Template
from prompttrail.core import Message, Session


class CalculationTemplate(Template):
    """Template for performing calculations"""

    def __init__(self, template_id: Optional[str] = None):
        super().__init__(template_id=template_id)

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        # Simulate calculation process
        yield Message(role="assistant", content="Let me solve this step by step:")
        yield Message(
            role="assistant", content="1. First, let's identify the key numbers..."
        )
        yield Message(role="assistant", content="The result is 42")
        return session

    def create_stack(self, session: Session) -> Stack:
        """Create stack frame for this template."""
        return Stack(template_id=self.template_id)


class MathProblemTemplate(Template):
    """Template for solving math problems using subroutines"""

    def __init__(self, template_id: Optional[str] = None):
        super().__init__(template_id=template_id)

        # Create calculation subroutine
        calculation = CalculationTemplate()
        self.calculation_subroutine = SubroutineTemplate(
            template=calculation,
            session_init_strategy=InheritSystemStrategy(),
            squash_strategy=FilterByRoleStrategy(roles=["assistant"]),
            template_id="calculation_subroutine",
        )

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        # Initial response
        yield Message(role="assistant", content="I'll help solve this math problem.")

        # Execute calculation subroutine
        for msg in self.calculation_subroutine.render(session):
            yield msg

        # Final summary
        yield Message(
            role="assistant",
            content="Problem solved! Let me know if you need any clarification.",
        )
        return session

    def create_stack(self, session: Session) -> Stack:
        """Create stack frame for this template."""
        return Stack(template_id=self.template_id)


def main():
    """Example usage of math problem solving with subroutines"""
    # Create session with system message
    session = Session()
    session.append(
        Message(
            role="system",
            content="You are a helpful math tutor who explains solutions step by step.",
        )
    )
    session.append(
        Message(
            role="user",
            content="Please solve this math problem: What is the answer to life, the universe, and everything?",
        )
    )

    # Create and run template
    solver = MathProblemTemplate()
    for message in solver.render(session):
        print(f"{message.role}: {message.content}")


if __name__ == "__main__":
    main()
