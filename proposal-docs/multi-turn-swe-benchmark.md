# Multi-Turn Software Engineering Benchmark

## Overview
A benchmark suite for evaluating LLM capabilities in realistic software engineering tasks using PromptTrail. Each test provides a sequence of steps in the initial prompt, and the LLM must complete them using available tools, following the pattern established in coding_agent.py.

## Core Components

### Test Definition

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BenchmarkTest:
    name: str
    description: str
    initial_prompt: str
    validation_command: str
    setup_commands: Optional[List[str]] = None

# Example Test
git_operations_test = BenchmarkTest(
    name="complex_git_operations",
    description="Test ability to perform complex git operations",
    initial_prompt="""Complete these git operations in sequence:
1. Create a new branch 'feature/test' from main
2. Cherry-pick commit abc123 from branch 'old-feature'
3. Resolve the merge conflict in file.txt
4. Rebase the changes onto main
5. Create a pull request""",
    validation_command="git log --graph --oneline | grep 'abc123'",
    setup_commands=[
        "git init",
        "git checkout -b main",
        "echo 'initial' > file.txt",
        "git add file.txt",
        "git commit -m 'initial commit'"
    ]
)
```

### Benchmark Runner

```python
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    LinearTemplate,
    ToolingTemplate,
    UserTemplate,
)
from prompttrail.agent.tools.builtin import (
    ExecuteCommand,
    ReadFile,
    CreateOrOverwriteFile,
    EditFile,
)
from prompttrail.core import Session
import docker

class BenchmarkRunner:
    def __init__(self, image: str = "python:3.11"):
        self.client = docker.from_client()
        self.image = image
        self.tools = [
            ExecuteCommand(),
            ReadFile(),
            CreateOrOverwriteFile(),
            EditFile(),
        ]
    
    def get_template(self, test: BenchmarkTest) -> LinearTemplate:
        return LinearTemplate([
            UserTemplate(
                content=f"""
Execute the following task: {test.description}

Steps:
{test.initial_prompt}

Rules:
- Use the provided tools for all actions
- Each step must be completed in sequence
- The task will be validated using: {test.validation_command}
                """
            ),
            ToolingTemplate(tools=self.tools)
        ])
    
    def run_test(self, test: BenchmarkTest, model) -> bool:
        # Create fresh container
        container = self.client.containers.run(
            self.image,
            detach=True,
            remove=True,
            working_dir="/workspace",
            volumes={
                "/tmp/benchmark": {"bind": "/workspace", "mode": "rw"}
            }
        )
        
        try:
            # Run setup commands
            if test.setup_commands:
                for cmd in test.setup_commands:
                    container.exec_run(cmd)
            
            # Initialize session
            session = Session(
                metadata={
                    "test_name": test.name,
                    "container_id": container.id,
                    "workspace": "/workspace"
                }
            )
            
            # Create runner
            runner = CommandLineRunner(
                model=model,
                template=self.get_template(test),
                tools=self.tools
            )
            
            # Run test
            runner.run(session=session)
            
            # Validate result
            result = container.exec_run(test.validation_command)
            return result.exit_code == 0
            
        finally:
            container.stop()
```

### Pytest Integration

```python
# test_benchmarks.py
import pytest
from prompttrail.models import AnthropicModel, OpenAIModel

TESTS = [
    git_operations_test,
    tdd_test,
    system_design_test
]

MODELS = [
    AnthropicModel(model_name="claude-3-5-sonnet-latest"),
    OpenAIModel(model_name="gpt-4")
]

@pytest.fixture
def benchmark_runner():
    return BenchmarkRunner()

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("test", TESTS)
def test_benchmark(benchmark_runner, model, test):
    assert benchmark_runner.run_test(test, model)
```

## Example Tests

### 1. Git Operations
Tests complex git workflow understanding and execution.

```python
git_operations_test = BenchmarkTest(
    name="git_operations",
    description="Complex git operations",
    initial_prompt="""
Complete these git operations in sequence:
1. Create feature branch
2. Cherry-pick specific commit
3. Resolve merge conflicts
4. Rebase onto main
5. Create pull request
    """,
    validation_command="git log --graph --oneline | grep 'abc123'"
)
```

### 2. Test-Driven Development
Tests ability to implement features based on failing tests.

```python
tdd_test = BenchmarkTest(
    name="tdd",
    description="Test-driven development",
    initial_prompt="""
Fix the failing test by implementing the required functionality:
1. Read test_calculator.py
2. Implement calculator.py
3. Run tests to verify
4. Add docstrings
5. Ensure all tests pass
    """,
    validation_command="python -m pytest test_calculator.py"
)
```

### 3. System Design
Tests ability to implement system components from specifications.

```python
system_design_test = BenchmarkTest(
    name="system_design",
    description="System implementation",
    initial_prompt="""
Implement an API service with these requirements:
1. Create FastAPI endpoints
2. Add error handling
3. Implement unit tests
4. Add API documentation
5. Ensure tests pass
    """,
    validation_command="python -m pytest test_api.py"
)
```

## Directory Structure

```
src/prompttrail/benchmark/
├── __init__.py
├── runner.py          # BenchmarkRunner implementation
├── tests/            # Test definitions
└── templates/        # Test scenario templates

tests/benchmark/      # Pytest test files
└── test_benchmarks.py

docker/              # Docker environment
└── Dockerfile
```

## Implementation Plan

1. Core Implementation (Week 1)
   - Create BenchmarkRunner
   - Implement test templates
   - Set up Docker environment

2. Test Creation (Week 2)
   - Implement Git Operations test
   - Implement TDD test
   - Create validation commands

3. Integration (Week 3)
   - Docker integration
   - Pytest setup
   - Documentation

## Next Steps

1. Create prototype in examples/benchmark/prototype.py
2. Set up base Docker image
3. Implement first test
4. Add CI pipeline