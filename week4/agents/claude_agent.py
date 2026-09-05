"""A Groq tool-calling agent with Tavily search and safe calculations.

Install dependencies first:
    pip install groq tavily-python python-dotenv

Set GROQ_API_KEY and TAVILY_API_KEY in python.env (or .env), then run:
    python claude_agent.py
"""

import ast
import json
import operator
import os

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


# `python.env` is the filename used by this project.  The second call also
# supports the conventional `.env` name if you choose to use it instead.
load_dotenv("python.env")
load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODEL = "openai/gpt-oss-120b"


def web_search(query: str) -> str:
    """Search the web with Tavily and return relevant, readable results."""
    try:
        result = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
    except Exception as error:
        return f"Web search failed: {error}"

    answer = result.get("answer")
    sources = result.get("results", [])
    formatted_sources = [
        f"- {item.get('title', 'Untitled')}: {item.get('content', '')}\n  Source: {item.get('url', '')}"
        for item in sources
    ]
    return "\n".join(
        part for part in [answer or "", *formatted_sources] if part
    ) or "No useful web-search results were found."


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_expression(node: ast.AST) -> int | float:
    """Evaluate only numeric arithmetic AST nodes; never execute Python code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_expression(node.left), _evaluate_expression(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_expression(node.operand))
    raise ValueError("Only numbers and +, -, *, /, //, %, **, and parentheses are allowed.")


def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression safely."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_evaluate_expression(tree.body))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
        return f"Calculation error: {error}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for current facts, news, definitions, or information not provided by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact web-search query to run.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, for example '(12 * 3) + 4'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A numeric expression using +, -, *, /, //, %, **, and parentheses.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS = {"web_search": web_search, "calculate": calculate}


def ask_agent(user_query: str) -> str:
    """Let the model decide whether it needs a tool, then return its final reply."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when useful. Cite source URLs returned by web_search.",
        },
        {"role": "user", "content": user_query},
    ]

    while True:
        response = groq.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content or "I could not generate an answer."

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
                result = TOOL_FUNCTIONS[name](**arguments)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                result = f"Tool error: {error}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result,
                }
            )


if __name__ == "__main__":
    query = input("Ask a question: ").strip()
    if query:
        print(ask_agent(query))
