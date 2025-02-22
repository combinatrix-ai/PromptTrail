# fmt: off
# flake8: noqa
# This is a snippet for the README.md. So, no formatting is needed.

# https://carbon.now.sh/?bg=rgba%28171%2C184%2C195%2C0%29&t=vscode&wt=none&l=python&width=680&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=0px&ph=0px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=4x&wm=false&code=%253E%2520%2523%2520%25F0%259F%25A4%2596%2520Multiple%2520LLM%2520models%2520are%2520supported%2520through%2520a%2520unified%2520API%250A%253E%2520model%2520%253D%2520AnthropicModel%28%250A%253E%2520%2520%2520configuration%253DAnthropicConfig%28%250A%253E%2520%2520%2520%2520%2520api_key%253Dos.environ%255B%2522ANTHROPIC_API_KEY%2522%255D%252C%250A%253E%2520%2520%2520%2520%2520model_name%253D%2522claude-3-5-sonnet-latest%2522%252C%250A%253E%2520%2520%2520%2520%2520max_tokens%253D4096%252C%250A%253E%2520%2520%2520%2520%2520tools%253D%255BExecuteCommand%28%29%252C%2520ReadFile%28%29%252C%2520CreateOrOverwriteFile%28%29%252C%2520EditFile%28%29%255D%29%29%250A%253E%250A%253E%2520%2523%2520%25F0%259F%258F%2597%25EF%25B8%258F%2520Create%2520your%2520agent%2520with%2520an%2520intuitive%2520Domain-Specific%2520Language%2520%28DSL%29%250A%253E%2520templates%2520%253D%2520LinearTemplate%28%255B%250A%253E%2520%2520%2520%2523%2520%25E2%259C%25A8%2520Generate%2520dynamic%2520messages%2520using%2520Jinja2%2520templating%250A%253E%2520%2520%2520SystemTemplate%28content%253D%250A%253E%2520%2520%2520%2520%2520%2522You%27re%2520a%2520smart%2520coding%2520agent%21%2520Type%2520END%2520if%2520you%2520want%2520to%2520end%2520conversation.%2520Follow%2520rules%253A%2520%257B%257Bclinerules%257D%257D%2522%29%252C%250A%253E%2520%2520%2520%2523%2520%25F0%259F%2594%2584%2520Supports%2520all%2520standard%2520control%2520flows%2520%28while%252Ffor%252C%2520if%252Felse%252C%2520functions%29%250A%253E%2520%2520%2520LoopTemplate%28%255B%250A%253E%2520%2520%2520%2520%2520%2523%2520%25F0%259F%2592%25AC%2520Handle%2520user%2520interactions%2520seamlessly%250A%253E%2520%2520%2520%2520%2520UserTemplate%28description%253D%2522Input%253A%2520%2522%29%252C%250A%253E%2520%2520%2520%2520%2520%2523%2520%25F0%259F%259B%25A0%25EF%25B8%258F%2520Integrate%2520powerful%2520built-in%2520tools%2520for%2520function%2520calling%2520and%2520automation%250A%253E%2520%2520%2520%2520%2520ToolingTemplate%28tools%253D%255BExecuteCommand%28%29%252CReadFile%28%29%252CCreateOrOverwriteFile%28%29%252CEditFile%28%29%255D%29%255D%252C%250A%253E%2520%2520%2520%2520%2520%2523%2520%25F0%259F%25A7%25A9%2520Easily%2520construct%2520complex%2520control%2520flow%2520logic%250A%253E%2520%2520%2520%2520%2520exit_condition%253Dlambda%2520session%253A%2520session.messages%255B-1%255D.content%2520%253D%253D%2520%2522END%2522%29%255D%29%250A%253E%250A%253E%2520%2523%2520%25F0%259F%2593%25A6%2520Use%2520metadata%2520to%2520efficiently%2520pass%252C%2520store%252C%2520and%2520retrieve%2520information%2520within%2520your%2520agent%250A%253E%2520initial_session%2520%253D%2520Session%28metadata%253D%257B%2522clinerules%2522%253A%2520open%28%2522.clinerules%2522%29.read%28%29%257D%29%250A%253E%250A%253E%2520%2523%2520%25F0%259F%259A%2580%2520Deploy%2520your%2520agent%2520anywhere%2520-%2520terminal%252C%2520server%252C%2520or%2520other%2520environments%250A%253E%2520runner%2520%253D%2520CommandLineRunner%28model%253Dmodel%252C%2520template%253Dtemplates%252C%2520user_interface%253DCLIInterface%28%29%29%250A%253E%2520runner.run%28session%253Dinitial_session%29%250A%250A%253D%253D%253D%253D%253D%2520Start%2520%253D%253D%253D%253D%253D%250AFrom%253A%2520%25F0%259F%2593%259D%2520system%250Amessage%253A%2520%2520%2522You%27re%2520a%2520smart%2520coding%2520agent%21%2520Type%2520END%2520if%2520you%2520want%2520to%2520end%2520conversation.%2520Follow%2520rules%253A%2520...%2522%250Ametadata%253A%2520%2520%257B%27clinerules%27%253A%2520%27...%27%257D%250A%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%253D%250AFrom%253A%2520%25F0%259F%2591%25A4%2520user%250AInput%253A%2520

import os

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import *
from prompttrail.agent.tools.builtin import *
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel

# 🤖 Multiple LLM models are supported through a unified API
model = AnthropicModel(
  AnthropicConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model_name="claude-3-5-sonnet-latest",
    max_tokens=4096,
    tools=[ExecuteCommand(), ReadFile(), CreateOrOverwriteFile(), EditFile()]))

# 🏗️ Create your agent with an intuitive Domain-Specific Language (DSL)
templates = LinearTemplate([
  # ✨ Generate dynamic messages using Jinja2 templating
  SystemTemplate(content=
    "You're a coding agent! Type END if you want to end conversation. Follow rules: {{clinerules}}"),
  # 🔄 Supports all standard control flows (while/for, if/else, functions)
  LoopTemplate([
    # 💬 Handle user interactions seamlessly
    UserTemplate(description="Input: "),
    # 🛠️ Integrate powerful built-in tools for function calling and automation
    ToolingTemplate(tools=[ExecuteCommand(),ReadFile(),CreateOrOverwriteFile(),EditFile()])],
    # 🧩 Easily construct complex control flow logic
    exit_condition=lambda session: session.messages[-1].content == "END")])

# 📦 Use metadata to efficiently pass, store, and retrieve information within your agent
initial_session = Session(metadata={"clinerules": open(".clinerules").read()})

# 🚀 Deploy your agent anywhere - terminal, server, or other environments
runner = CommandLineRunner(model=model, template=templates, user_interface=CLIInterface())
runner.run(session=initial_session)

# ===== Start =====
# From: 📝 system
# message:  "You're a smart coding agent! Type END if you want to end conversation. Follow rules: ..."
# metadata:  {'clinerules': '...'}
# =================
# From: 👤 user
# Input:
