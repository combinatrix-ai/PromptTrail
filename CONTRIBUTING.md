# Contributing Guideline

Welcome to the PromptTrail project! We're so glad you're thinking about contributing to our project!

## Development Setup

- We use [rye](https://rye-up.com/) to manage our Python development environment.
- You can run an example as follows:
  - Installing rye (See their [website](https://rye-up.com/) for the latest installation instruction)
    - `curl -sSf https://rye-up.com/get | bash`
  - Clone this repository where you want
    - `git clone https://github.com/combinatrix-ai/PromptTrail.git`
  - Setup the environment
    - `cd PromptTrail`
    - Install dependencies
      - `rye sync`
    - Activate the environment
      - `rye shell`
  - Run an example
    - `python -m examples.example_openai.py`
      - You may need to set `OPENAI_API_KEY` environment variable to run this example.

## Before you make PRs

- Please make sure that your code passes all tests. (adding tests is always welcome!)
  - `rye run test`
- Please make sure that your code is auto-formatted. (black, autoflake, isort)
  - `rye run format`
- Please make sure that your code passes the lint check. (mypy)
  - `rye run lint`
- Please build documentation if you change the public interface. (sphinx)
  - `rye run doc`

## License

- This project is currently licensed under the [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license).
  - If you're fine with Elasticsearch, then you may be also fine with this project.
  - Elastic License 2.0 is not OSI-compliant, but if you're OK with Llama 2 License, then you may be also OK to use non-OSI-compliant license.
- We choose the license because we plan to provide a web service to let everyone including non-developers to collaboratively create, edit, test, and host templates and agent online.
  - We want to build Huggingface / Github for LLM programming!
  - If you want to build this kind of service, we hope you contact us first if you want to use this project.
- In other use cases, which means almost everything you may do, you can use this project commercially.

- Therefore, before you contribute, you must agree that your contribution will be licensed under the [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license) or other licenses we have chosen for this project.
  - You grant to us a worldwide, royalty-free, exclusive, perpetual and irrevocable license, with the right to transfer an unlimited number of non-exclusive licenses or to grant sublicenses to third parties, under the Copyright covering the Contribution to use the Contribution by all means, including, but not limited to:
    - publish the Contribution
    - modify the Contribution
    - prepare derivative works based upon or containing the Contribution and/or to combine the Contribution with other Materials
    - reproduce the Contribution in original or modified form
    - distribute, to make the Contribution available to the public, display and publicly perform the Contribution in original or modified form

- The senteces above may seem a little bit scary, but in essence, we want you to declare that your contribution can be used freely in this project, so that other people can use it without worry.

- Note: Elastic License 2.0 is not compatible with GPL, so we may not use GPL libraries in this project.
