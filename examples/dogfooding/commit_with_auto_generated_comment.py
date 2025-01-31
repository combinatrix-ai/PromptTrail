import os
import subprocess
import tempfile
from typing import Optional

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.session_transformers import (
    MetadataTransformer,
    ResetMetadata,
    SessionTransformer,
)
from prompttrail.agent.templates import (
    AssistantTemplate,
    BreakTemplate,
    IfTemplate,
    LinearTemplate,
    LoopTemplate,
    MessageTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Metadata, Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel


class RewriteMessage(SessionTransformer):
    def __init__(self, index: int, new_content: str):
        self.index = index
        self.new_content = new_content

    def process(self, session: Session) -> Session:
        session.messages[self.index].content = self.new_content
        return session


class SaveCommitMessage(MetadataTransformer):
    """Hook to save the generated commit message in the state data."""

    def process_metadata(self, metadata: Metadata, session: Session) -> Metadata:
        commit_message = session.get_last_message().content
        metadata["commit_message"] = commit_message
        return metadata


def get_git_info() -> dict:
    """
    Retrieve Git repository information including staged changes, current branch, and recent commit history.

    Returns
    -------
    dict
        A dictionary containing:
        - 'diff': Staged changes (git diff --cached)
        - 'branch': Current branch name
        - 'log': Recent commit history (last 5 commits)
    """
    try:
        # Get staged changes
        diff = subprocess.check_output(
            ["git", "diff", "--cached"], text=True, stderr=subprocess.PIPE
        )

        # Get current branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

        # Get recent commit history (last 5 commits)
        log = subprocess.check_output(
            ["git", "log", "-5", "--oneline"], text=True, stderr=subprocess.PIPE
        )

        return {"diff": diff, "branch": branch, "log": log}
    except subprocess.CalledProcessError as e:
        print(
            f"Warning: Git command failed: {e.stderr.decode() if e.stderr else str(e)}"
        )
        return {"diff": "", "branch": "unknown", "log": ""}


def execute_git_commit(message: str) -> bool:
    """
    Create a Git commit with the provided message.

    Parameters
    ----------
    message : str
        The commit message to use

    Returns
    -------
    bool
        True if the commit was successful, False otherwise
    """
    try:
        # Store the commit message in a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(message)
            temp_path = f.name

        try:
            # Create the commit using the temporary file
            subprocess.check_output(
                ["git", "commit", "-F", temp_path], text=True, stderr=subprocess.PIPE
            )
            print("Changes committed successfully!")
            return True
        finally:
            # Remove the temporary file
            os.unlink(temp_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create commit: {e.stderr.decode() if e.stderr else str(e)}")
        return False


templates = LinearTemplate(
    [
        SystemTemplate(
            template_id="instruction",
            content="""
You are an expert at crafting Git commit messages.
Your task is to analyze the provided Git repository information and generate an appropriate commit message.

Commit message format:
- Line 1: Summary of changes (maximum 50 characters)
- Line 2: Blank line
- Line 3+: Detailed explanation of changes (when necessary)
- Final line: (This commit message is auto-generated with examples/dogfooding/commit_with_auto_generated_comment.py)

Provide only the commit message without any additional commentary or formatting.
""",
        ),
        MessageTemplate(
            template_id="git_info",
            role="user",
            content="""
Please generate a commit message based on the following information:

1. Current branch: {{branch}}
2. Staged changes (git diff --cached):
{{diff}}
3. Recent commit history:
{{log}}
""",
            after_transform=ResetMetadata(),
        ),
        LoopTemplate(
            [
                AssistantTemplate(
                    template_id="generate_commit_message",
                    after_transform=SaveCommitMessage(),
                ),
                UserTemplate(
                    template_id="get_feedback",
                    description="Please provide your feedback:",
                    default="Looks good!",
                ),
                MessageTemplate(
                    template_id="analyze_feedback",
                    role="assistant",
                    content="""
Analyzing the user's feedback to determine the next action:

1. If the feedback includes ANY suggestions for improvement (even minor ones), respond with:
RETRY

2. If the feedback is positive with NO suggestions for changes, respond with:
END

** Response must be either "RETRY" or "END" based on the feedback analysis. **

Examples:
- Respond with "RETRY" for repsonses like "not good", "Good, but add an emoji", "Great, but it would be better with xxx"
- Respond with "END" for responses like "Perfect!", "OK", "Thanks"
""",
                ),
                AssistantTemplate(
                    template_id="feedback_decision",
                ),
                IfTemplate(
                    true_template=BreakTemplate(),
                    false_template=AssistantTemplate(
                        before_transform=RewriteMessage(-1, "RETRY"),
                        content="Based on your feedback, I will regenerate the commit message.",
                    ),
                    condition=lambda session: "END"
                    in session.get_last_message().content,
                ),
            ],
        ),
    ]
)


def main(
    api_key: Optional[str] = None,
    execute_commit: bool = True,
) -> str:
    """
    Generate a commit message and optionally create a Git commit.

    Parameters
    ----------
    api_key : Optional[str]
        Anthropic API key. If not provided, will be read from the ANTHROPIC_API_KEY environment variable
    execute_commit : bool
        Whether to create a Git commit after generating the message (default: True)

    Returns
    -------
    str
        The generated commit message
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    configuration = AnthropicConfig(
        api_key=api_key,
        model_name="claude-3-5-sonnet-latest",
        temperature=0.7,
        max_tokens=1000,
    )
    model = AnthropicModel(configuration=configuration)

    runner = CommandLineRunner(
        model=model,
        template=templates,
        user_interface=CLIInterface(),
    )

    git_info = get_git_info()
    if not any(git_info.values()):
        print("Warning: No Git information found. Are you in a Git repository?")
        return ""

    initial_session = Session(
        metadata={
            "branch": git_info["branch"],
            "diff": git_info["diff"],
            "log": git_info["log"],
        }
    )

    session = runner.run(session=initial_session)

    return session.metadata["commit_message"]


if __name__ == "__main__":
    commit_message = main()
    if commit_message:
        print("\nGenerated commit message:\n")
        print(commit_message)

    # Confirm with user before creating the commit
    while True:
        user_input = input("\nWould you like to create this commit? (y/n): ")
        if user_input.lower() == "y":
            execute_git_commit(commit_message)
            break
        elif user_input.lower() == "n":
            print("No commit was created.")
            break
        else:
            print("Please enter either 'y' or 'n'.")
