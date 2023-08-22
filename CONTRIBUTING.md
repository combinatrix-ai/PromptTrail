# Contributing Guideline

Welcome to the PromptTrail project! We're glad you're considering contributing to our project!

## Development Setup

- We use [rye](https://rye-up.com/) to manage our Python development environment.
- You can set up your development environment as follows:
  - Install rye (See their [website](https://rye-up.com/) for the latest installation instructions)
    - `curl -sSf https://rye-up.com/get | bash`
  - Clone this repository to your desired location
    - `git clone https://github.com/combinatrix-ai/PromptTrail.git`
  - Set up the environment
    - `cd PromptTrail`
    - Install dependencies
      - `rye sync`
    - Activate the environment
      - `rye shell`
  - Run an example
    - `python -m examples.example_openai.py`
      - You may need to set the `OPENAI_API_KEY` environment variable to run this example.

## Before Submitting Pull Requests

- Please ensure that your code passes all tests. (Adding tests is always welcome!)
  - `rye run test`
- Please ensure that your code is auto-formatted. (black, autoflake, isort)
  - `rye run format`
- Please ensure that your code passes the lint check. (mypy)
  - `rye run lint`
- Please build the documentation if you change the public interface. (sphinx)
  - `rye run doc`

Note: You can run all of these commands at once using `rye run all`. GitHub Actions will automatically run these commands in check mode when you submit a pull request. If any of these checks fail, please run these commands locally and fix any errors.

## License

- This project is currently licensed under the [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license).
  - If you're comfortable with Elasticsearch, you should be comfortable with this project as well.
  - The Elastic License 2.0 is not OSI-compliant, but if you're okay with the Llama 2 License, you should be okay to use a non-OSI-compliant license.
- We chose this license because we plan to provide a web service that allows everyone, including non-developers, to collaboratively create, edit, test, and host templates and agents online.
  - We aim to build a Huggingface/Github for LLM programming!
  - If you want to build this kind of service, we hope you'll contact us first if you want to use this project.
- In other use cases, which means almost everything you might do, you can use this project commercially.

- Therefore, before you contribute, you must agree that your contribution will be licensed under the [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license) or other licenses we have chosen for this project.
  - You grant us a worldwide, royalty-free, exclusive, perpetual, and irrevocable license, with the right to transfer an unlimited number of non-exclusive licenses or to grant sublicenses to third parties, under the Copyright covering the Contribution to use the Contribution by all means, including, but not limited to:
    - Publishing the Contribution
    - Modifying the Contribution
    - Preparing derivative works based upon or containing the Contribution and/or combining the Contribution with other Materials
    - Reproducing the Contribution in its original or modified form
    - Distributing, making the Contribution available to the public, displaying, and publicly performing the Contribution in its original or modified form

- The sentences above may seem a little bit intimidating, but in essence, we want you to declare that your contribution can be used freely in this project so that other people can use it without worry.

- Note: The Elastic License 2.0 is not compatible with GPL, so we may not use GPL libraries in this project.

## Coding Principles

- If you know what an LLM is, you must be able to use PromptTrail.
- Agent (Flow) as Code
  - Agent that can be written in one place by code
    - Hook-based agent definition like PyTorch Lightning
- Provide an easy way to debug prompt program
  - Record everything for later inspection
  - Easy to read error messages with template id, hook name, etc... is included
- Intuitive and explicit (but sometimes convention)
  - Everything evolves fast here. You can't be sure what is right now. So explicit is better than implicit. Code is better than document.
    - No hidden templates and configurations
    - Every parameter should be passed explicitly and be able to be understood by types
      - Easy to work with on VSCode and JetBrains IDEs
  - Everything must be clear by class inheritance and types. I don't want to read docs.
    - Unified access to templates, parameters of agents
    - Hook-based agent definition
    - More default values

### Type Checking

- We use pylance (strict) in VSCode and mypy.
  - We prefer repeating duplicated type checking code over using `Any` or `# type: ignore`.
