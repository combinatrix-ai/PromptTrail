# Prompttrail Examples

In this directory, we collect examples of `prompttrail`.

## `provider`

In this directory, we give examples of how to use `prompttrail` as a thin wrapper around LLMs.
If you want just to use LLMs, see these examples.

- [`provider/openai.py`](provider/openai.py)
  - This shows how to use `prompttrail` to generate prompts for OpenAI's GPT-3.5 / 4.
- [`provider/stream.py`](provider/stream.py)
  - If you want to use OpenAI's streaming feature, see this example.
- [`provider/google.py`](provider/google.py)
  - This shows how to use `prompttrail` to generate prompts for Google's Gemini.
- [`provider/mock.py`](provider/mock.py)
  - This shows how you can mock APIs for testing.

## `agent`

In this directory, we give examples of how to use `prompttrail` to create agents.
If you want to use complex logic, function calling and so on, see these examples.

- [`agent/fermi_problem.py`](agent/fermi_problem.py)
  - This shows how to use `prompttrail` to create an simple agent which take multiple turns, call functions, and so on.
  - You can also see how you can mock the APIs for testing.
- [`agent/weather_forecast.py`](agent/weather_forecast.py)
  - This shows how to use `prompttrail` to create an agent which use function calling feature of `prompttrail`.

## `dogfooding`

In this directory, we give examples of how to use `prompttrail` as we actually use in this directory.

- [`dogfooding/create_unit_test.py`](dogfooding/create_unit_test.py)
  - This shows how to use `prompttrail` to create unit tests for `prompttrail` itself.
  - You can see the result in this [PR](https://github.com/combinatrix-ai/PromptTrail/pull/4)
- [`dogfooding/fix_markdown.py`](dogfooding/fix_markdown.py)
  - This shows how to use `prompttrail` to fix markdown files in this repository.
  - You can see the result in this [PR](https://github.com/combinatrix-ai/PromptTrail/pull/3)
