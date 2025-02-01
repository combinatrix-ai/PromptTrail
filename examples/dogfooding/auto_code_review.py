#!/usr/bin/env python3

"""
An automated code review script that uses LLM to review code changes in pull requests
and posts review comments using GitHub's API.

See .github/workflows/auto-code-review.yml for the GitHub Actions workflow
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict, cast

import requests

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    AssistantTemplate,
    LinearTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel


class PRFile(TypedDict):
    """Type definition for a PR file."""

    filename: str
    content: str
    patch: str


class ReviewComment(TypedDict):
    """Type definition for a review comment."""

    path: str
    line: int
    body: str


class GitHubReviewComment(TypedDict):
    """Type definition for a GitHub API review comment."""

    body: str
    commit_id: str
    path: str
    line: int
    side: str


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Bot configuration from GitHub Actions workflow
BOT_NAME = os.getenv("GITHUB_WORKFLOW_NAME", "Automated Code Review")


def delete_review_comments(
    token: str,
    owner: str,
    repo: str,
    pr_number: int,
) -> None:
    """
    Delete all review comments on the pull request.

    Parameters
    ----------
    token : str
        GitHub API token
    owner : str
        Repository owner
    repo : str
        Repository name
    pr_number : int
        Pull request number
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Get all review comments on the PR
    # Note: This endpoint only returns review comments, not regular PR comments
    comments_url = (
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    )
    response = requests.get(comments_url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Error fetching review comments: {response.text}")
        return

    try:
        comments = response.json()
        logger.debug(f"Found {len(comments)} review comments")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing comments response: {e}")
        return

    # Delete comments from our bot
    deleted_count = 0
    for comment in comments:
        comment_id = comment.get("id")
        body = comment.get("body", "")

        logger.debug(f"Checking comment {comment_id}:")
        logger.debug(f"  Body: {body}")

        # Delete if it's a comment from our bot
        if comment_id and f"[{BOT_NAME}]" in body:
            logger.debug(f"Attempting to delete review comment {comment_id}")
            delete_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/comments/{comment_id}"
            delete_resp = requests.delete(delete_url, headers=headers)

            if delete_resp.status_code == 204:
                deleted_count += 1
                logger.info(f"Deleted review comment {comment_id}")
            else:
                logger.warning(
                    f"Failed to delete review comment {comment_id}: "
                    f"Status {delete_resp.status_code}, Response: {delete_resp.text}"
                )

    # Get all reviews
    reviews_url = (
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    )
    response = requests.get(reviews_url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Error fetching reviews: {response.text}")
        return

    try:
        reviews = response.json()
        logger.debug(f"Found {len(reviews)} reviews")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing reviews response: {e}")
        return

    # Delete reviews from our bot
    for review in reviews:
        review_id = review.get("id")
        body = review.get("body", "")

        logger.debug(f"Checking review {review_id}:")
        logger.debug(f"  Body: {body}")

        # Delete if it's a review from our bot
        if review_id and f"[{BOT_NAME}]" in body:
            logger.debug(f"Attempting to delete review {review_id}")
            delete_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}"
            delete_resp = requests.delete(delete_url, headers=headers)

            if delete_resp.status_code == 204:
                deleted_count += 1
                logger.info(f"Deleted review {review_id}")
            else:
                logger.warning(
                    f"Failed to delete review {review_id}: "
                    f"Status {delete_resp.status_code}, Response: {delete_resp.text}"
                )

    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} review comments and reviews")
    else:
        logger.info("No review comments or reviews found to delete")


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON array from text that might contain additional content.

    Parameters
    ----------
    text : str
        Text that might contain JSON array

    Returns
    -------
    Optional[str]
        Extracted JSON string if found, None otherwise
    """
    # Try to find JSON array pattern
    matches = re.findall(r"\[[\s\S]*\]", text)
    if matches:
        # Return the first match that is valid JSON
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

    # Try to find the content between ```json and ``` if exists
    matches = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    if matches:
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

    return None


def fetch_pr_files(token: str, owner: str, repo: str, pr_number: int) -> List[PRFile]:
    """
    Fetch changed files from a GitHub pull request.

    Parameters
    ----------
    token : str
        GitHub API token
    owner : str
        Repository owner
    repo : str
        Repository name
    pr_number : int
        Pull request number

    Returns
    -------
    List[PRFile]
        List of changed files with their content
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    # Get list of changed files
    files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    response = requests.get(files_url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Error fetching changed files: {response.text}")
        sys.exit(1)

    try:
        files_changed = response.json()
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing files response: {e}")
        sys.exit(1)

    pr_files: List[PRFile] = []

    for file in files_changed:
        filename = file.get("filename")
        raw_url = file.get("raw_url")
        if not filename or not raw_url or not filename.endswith(".py"):
            continue

        # Fetch file content
        file_resp = requests.get(raw_url)
        if file_resp.status_code != 200:
            logger.warning(f"Error fetching file {filename}")
            continue

        # Add line numbers to the content
        content_with_lines = []
        for i, line in enumerate(file_resp.text.splitlines(), start=1):
            content_with_lines.append(f"{i:4d} | {line}")

        pr_files.append(
            cast(
                PRFile,
                {
                    "filename": filename,
                    "content": "\n".join(content_with_lines),
                    "patch": file.get("patch", ""),
                },
            )
        )

    return pr_files


def post_review(
    token: str,
    owner: str,
    repo: str,
    pr_number: int,
    commit_id: str,
    review_comments: List[ReviewComment],
    pr_files: List[PRFile],
) -> None:
    """
    Post review comments to GitHub pull request.

    Parameters
    ----------
    token : str
        GitHub API token
    owner : str
        Repository owner
    repo : str
        Repository name
    pr_number : int
        Pull request number
    commit_id : str
        Commit SHA
    review_comments : List[ReviewComment]
        List of review comments
    pr_files : List[PRFile]
        List of PR files with their patches
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Format comments according to GitHub API requirements
    formatted_comments: List[GitHubReviewComment] = []
    for comment in review_comments:
        formatted_comment: GitHubReviewComment = {
            "body": f"[{BOT_NAME}]\n\n{comment['body']}",
            "commit_id": commit_id,
            "path": comment["path"],
            "line": comment["line"],
            "side": "RIGHT",  # Comment on the new version of the code
        }
        formatted_comments.append(formatted_comment)

    logger.debug(
        "Formatted review comments: %s", json.dumps(formatted_comments, indent=2)
    )

    # Post each comment individually as recommended by GitHub API
    for comment in formatted_comments:
        comment_url = (
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        )
        comment_resp = requests.post(comment_url, headers=headers, json=comment)

        if comment_resp.status_code not in (200, 201):
            logger.error(
                "Error posting comment. Status code: %d", comment_resp.status_code
            )
            logger.error("Response: %s", comment_resp.text)
            logger.error("Comment data: %s", json.dumps(comment, indent=2))
        else:
            logger.info(
                "Comment posted successfully for line %d in %s",
                comment["line"],
                comment["path"],
            )

    # Post a summary review
    review_url = (
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    )
    review_payload = {
        "commit_id": commit_id,
        "body": f"[{BOT_NAME}]\n\nCode review completed. Feedback has been provided as inline comments.\nNote: Previous review comments will be removed on subsequent runs to avoid duplicate feedback.",
        "event": "COMMENT",
    }

    review_resp = requests.post(review_url, headers=headers, json=review_payload)
    if review_resp.status_code not in (200, 201):
        logger.error(
            "Error posting review summary. Status code: %d", review_resp.status_code
        )
        logger.error("Response: %s", review_resp.text)
    else:
        logger.info("Review summary posted successfully")


def parse_review_comments(raw_comments: List[Dict[str, Any]]) -> List[ReviewComment]:
    """
    Parse and validate review comments from LLM response.

    Parameters
    ----------
    raw_comments : List[Dict[str, Any]]
        Raw comments from LLM response

    Returns
    -------
    List[ReviewComment]
        Validated review comments
    """
    validated_comments: List[ReviewComment] = []

    for comment in raw_comments:
        try:
            # Ensure required fields are present
            if not all(k in comment for k in ("path", "line", "body")):
                logger.warning("Comment missing required fields: %s", comment)
                continue

            # Convert line to int
            line = int(comment["line"])

            # Create ReviewComment with explicit type cast
            validated_comments.append(
                cast(
                    ReviewComment,
                    {
                        "path": str(comment["path"]),
                        "line": line,
                        "body": str(comment["body"]),
                    },
                )
            )
        except (ValueError, TypeError) as e:
            logger.warning("Error parsing comment: %s - %s", comment, e)
            continue

    return validated_comments


def main():
    """Main function to run the automated code review."""
    # Get GitHub environment variables
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("GITHUB_TOKEN not set")
        sys.exit(1)

    github_repository = os.getenv("GITHUB_REPOSITORY")
    if not github_repository or "/" not in github_repository:
        logger.error("GITHUB_REPOSITORY not set or invalid")
        sys.exit(1)
    owner, repo = github_repository.split("/")

    # Get pull request information
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path or not os.path.exists(event_path):
        logger.error("GITHUB_EVENT_PATH not set or file does not exist")
        sys.exit(1)
    with open(event_path, "r") as f:
        try:
            event_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing event data: {e}")
            sys.exit(1)

    pr = event_data.get("pull_request")
    if not pr:
        logger.error(
            "No pull_request info in event payload. Event data: %s",
            json.dumps(event_data, indent=2),
        )
        sys.exit(1)
    pr_number = pr.get("number")
    commit_id = pr.get("head", {}).get("sha")
    if not pr_number or not commit_id:
        logger.error(
            "Missing PR number or commit id. PR data: %s", json.dumps(pr, indent=2)
        )
        sys.exit(1)

    logger.info(f"Reviewing PR #{pr_number} in repository {owner}/{repo}")

    # Delete existing review comments
    delete_review_comments(github_token, owner, repo, pr_number)

    # Fetch changed files
    pr_files = fetch_pr_files(github_token, owner, repo, pr_number)
    if not pr_files:
        logger.info("No Python files to review.")
        sys.exit(0)

    # Create review template
    review_template = LinearTemplate(
        [
            SystemTemplate(
                content="""
You are an experienced software engineer performing a code review on Python code changes.
Focus on the following aspects:

1. Code Quality
   - Clean code principles
   - Proper naming conventions
   - Code duplication
   - Readability and maintainability

2. Performance
   - Efficient algorithms and data structures
   - Memory usage
   - Time complexity

3. Security
   - Potential vulnerabilities
   - Proper handling of sensitive data
   - Input validation

4. Testing
   - Code testability
   - Edge cases
   - Error scenarios

5. Best Practices
   - Python best practices
   - Error handling
   - Logging practices
   - Type hints usage

The code will be provided with line numbers in the format:
   1 | def example():
   2 |     return True

For each issue, provide:
- File name and line number (extract from the line number prefix)
- Issue description
- Suggested improvement

Format your response as a JSON array of review comments, each in this format:
{
    "path": "file_path",
    "line": line_number,
    "body": "detailed comment"
}

Do not include any additional text before or after the JSON array.
Also consider the patch information to focus on changed lines.
Prioritize significant issues over minor ones.
"""
            ),
            UserTemplate(
                content="""Review the following code changes:

{% for file in files %}
File: {{ file.filename }}
Patch:
{{ file.patch }}

Full file with line numbers:
```python
{{ file.content }}
```

{% endfor %}
"""
            ),
            AssistantTemplate(),
        ]
    )

    # Initialize model and runner
    model = AnthropicModel(
        configuration=AnthropicConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
            max_tokens=1024,
        )
    )

    runner = CommandLineRunner(
        model=model,
        template=review_template,
        user_interface=CLIInterface(),
    )

    # Run the review
    initial_session = Session(metadata={"files": pr_files})
    review_session = runner.run(session=initial_session)

    # Parse review comments
    review_response = review_session.get_last_message().content
    logger.debug("LLM response: %s", review_response)

    # Try to extract JSON from the response
    json_str = extract_json_from_text(review_response)
    if json_str is None:
        logger.error("Could not find JSON array in response")
        logger.error("Raw response: %s", review_response)
        sys.exit(1)

    try:
        raw_comments = json.loads(json_str)
        if not isinstance(raw_comments, list):
            logger.error(
                "Invalid review comments format. Expected list but got: %s",
                type(raw_comments),
            )
            sys.exit(1)

        # Parse and validate comments
        review_comments = parse_review_comments(raw_comments)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse review comments: %s", str(e))
        logger.error("Extracted JSON string: %s", json_str)
        sys.exit(1)

    if not review_comments:
        logger.info("No issues found during automated code review.")
        sys.exit(0)

    logger.info("Found %d issues to report", len(review_comments))

    # Post review comments
    post_review(
        github_token, owner, repo, pr_number, commit_id, review_comments, pr_files
    )
    logger.info("Automated code review posted successfully.")


if __name__ == "__main__":
    main()
