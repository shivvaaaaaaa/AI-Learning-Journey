import os
import random
from typing import Annotated, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

from pydantic import BaseModel, Field


# ============================================================
# ENV
# ============================================================

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# MENU
# ============================================================
# Change this later according to your actual menu.

MENU = {
    "pizza": 5,
    "burger": 10,
    "pasta": 3,
    "biryani": 7,
    "momos": 8,
}


# ============================================================
# ORDER EXTRACTION SCHEMA
# ============================================================

class OrderExtraction(BaseModel):

    is_food_order: bool = Field(
        description="True if the user is trying to order food."
    )

    dish: Optional[str] = Field(
        default=None,
        description="The dish the user wants to order."
    )

    quantity: Optional[int] = Field(
        default=None,
        description="The quantity the user wants."
    )


structured_llm = llm.with_structured_output(OrderExtraction)


# ============================================================
# STATE
# ============================================================

class OrderState(TypedDict):

    # Conversation between user and AI
    messages: Annotated[list[BaseMessage], add_messages]

    # Order details
    dish: str
    required_quantity: int

    # Must be string according to requirement
    available_quantity: str

    # Current status
    status: str

    # Remaining order retries
    order_retry_attempts: int

    # Remaining cook retries
    # Initial cook is NOT counted here.
    # Value 2 = two additional cook attempts available.
    cook_retry_attempts: int

    # Remaining serve retries
    # Initial serve is NOT counted here.
    serve_retry_attempts: int

    # Final result
    final_result: str


# ============================================================
# NODE 1
# EXTRACT ORDER
# ============================================================

def extract_order(state: OrderState):

    latest_message = state["messages"][-1]

    user_input = latest_message.content

    result = structured_llm.invoke(
        f"""
You are an AI agent ONLY for restaurant food ordering.

Your job is to understand whether the user's message
is related to ordering food.

If the user wants to order food:
- Extract the dish.
- Extract the quantity.

If the user asks something unrelated to food ordering:
- is_food_order = false
- dish = null
- quantity = null

Do not behave like a general-purpose assistant.

User message:
{user_input}
"""
    )

    # --------------------------------------------------------
    # NOT A FOOD ORDER
    # --------------------------------------------------------

    if not result.is_food_order:

        return {
            "status": "invalid",

            "messages": [
                AIMessage(
                    content=(
                        "I am a food ordering AI agent. "
                        "I can only help you with restaurant food orders."
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # INVALID ORDER DETAILS
    # --------------------------------------------------------

    if (
        not result.dish
        or result.quantity is None
        or result.quantity <= 0
    ):

        return {
            "status": "invalid",

            "messages": [
                AIMessage(
                    content=(
                        "Please provide a valid dish name and "
                        "a positive quantity."
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # VALID ORDER
    # --------------------------------------------------------

    return {
        "dish": result.dish.lower().strip(),
        "required_quantity": result.quantity,
        "status": "order_extracted"
    }


# ============================================================
# ROUTER AFTER EXTRACT ORDER
# ============================================================

def route_after_extract(state: OrderState):

    if state["status"] == "invalid":
        return "end"

    return "order_confirm"


# ============================================================
# NODE 2
# ORDER CONFIRM
# ============================================================

def order_confirm(state: OrderState):

    dish = state["dish"]
    required_quantity = state["required_quantity"]

    # If dish is not in menu → 0
    available = MENU.get(dish, 0)

    # Requirement says available_quantity is STRING
    available_quantity = str(available)

    # --------------------------------------------------------
    # CASE 1: NOT AVAILABLE
    # --------------------------------------------------------

    if available == 0:

        return {
            "available_quantity": available_quantity,
            "status": "unavailable",

            "messages": [
                AIMessage(
                    content=(
                        f"Sorry, {dish} is not available. "
                        f"Available quantity is 0."
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # CASE 2: FULLY AVAILABLE
    # --------------------------------------------------------

    if available >= required_quantity:

        return {
            "available_quantity": available_quantity,
            "status": "confirmed",

            "messages": [
                AIMessage(
                    content=(
                        f"{dish} is available. "
                        f"You requested {required_quantity} "
                        f"and {available} are available."
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # CASE 3: PARTIALLY AVAILABLE
    # --------------------------------------------------------

    return {
        "available_quantity": available_quantity,
        "status": "partial",

        "messages": [
            AIMessage(
                content=(
                    f"Only {available} {dish} are available, "
                    f"but you requested {required_quantity}."
                )
            )
        ]
    }


# ============================================================
# ROUTER AFTER ORDER CONFIRM
# ============================================================

def route_after_order_confirm(state: OrderState):

    if state["status"] == "confirmed":
        return "cook"

    if state["status"] in ["partial", "unavailable"]:
        return "order_decision"

    return "end"


# ============================================================
# NODE 3
# ORDER DECISION
# ============================================================

def order_decision(state: OrderState):

    print("\n--------------------------------")
    print("ORDER DECISION")
    print("--------------------------------")

    # ========================================================
    # PARTIAL ORDER
    # ========================================================

    if state["status"] == "partial":

        available = int(state["available_quantity"])

        print(
            f"\nOnly {available} {state['dish']} "
            f"are available."
        )

        print("\n1. Confirm partial order")
        print("2. Reject and place a new order")

        choice = input("\nEnter choice: ").strip()

        # ----------------------------------------------------
        # USER ACCEPTS PARTIAL
        # ----------------------------------------------------

        if choice == "1":

            return {
                "required_quantity": available,
                "status": "confirmed",

                "messages": [
                    HumanMessage(
                        content="I confirm the partial order."
                    ),

                    AIMessage(
                        content=(
                            f"Okay. I will prepare "
                            f"{available} {state['dish']}."
                        )
                    )
                ]
            }

        # ----------------------------------------------------
        # USER REJECTS PARTIAL
        # ----------------------------------------------------

        return {
            "status": "new_order",

            "order_retry_attempts": (
                state["order_retry_attempts"] - 1
            ),

            "messages": [
                HumanMessage(
                    content="I reject the partial order."
                ),

                AIMessage(
                    content=(
                        "Okay. Please provide a new order."
                    )
                )
            ]
        }

    # ========================================================
    # UNAVAILABLE ORDER
    # ========================================================

    print(
        f"\n{state['dish']} is currently unavailable."
    )

    print("\n1. Place a new order")
    print("2. Cancel order")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":

        return {
            "status": "new_order",

            "order_retry_attempts": (
                state["order_retry_attempts"] - 1
            ),

            "messages": [
                HumanMessage(
                    content="I want to place a new order."
                ),

                AIMessage(
                    content=(
                        "Sure. Please provide your new order."
                    )
                )
            ]
        }

    return {
        "status": "failed",

        "final_result": (
            "Order cancelled by the user."
        )
    }


# ============================================================
# ROUTER AFTER ORDER DECISION
# ============================================================

def route_after_order_decision(state: OrderState):

    # Partial order accepted
    if state["status"] == "confirmed":
        return "cook"

    # User wants a new order
    if state["status"] == "new_order":

        # No more order retries
        if state["order_retry_attempts"] <= 0:
            return "end"

        return "new_order_input"

    return "end"


# ============================================================
# NODE 4
# NEW ORDER INPUT
# ============================================================

def new_order_input(state: OrderState):

    print("\n--------------------------------")
    print("NEW ORDER")
    print("--------------------------------")

    user_input = input(
        "\nEnter your new order: "
    ).strip()

    return {
        "messages": [
            HumanMessage(content=user_input)
        ],

        "status": "new_order_input"
    }


# ============================================================
# NODE 5
# COOK
# ============================================================

def cook(state: OrderState):

    print("\n--------------------------------")
    print("COOK")
    print("--------------------------------")

    # --------------------------------------------------------
    # Every call to cook is an actual attempt.
    #
    # cook_retry_attempts represents ADDITIONAL attempts
    # available after the current cook call.
    # --------------------------------------------------------

    print(
        "Remaining cook retries before this attempt:",
        state["cook_retry_attempts"]
    )

    # 60% SUCCESS
    # 40% FAILURE

    success = random.random() < 0.60

    # --------------------------------------------------------
    # COOK SUCCESS
    # --------------------------------------------------------

    if success:

        print("COOK → SUCCESS")

        return {
            "status": "ready",

            "messages": [
                AIMessage(
                    content=(
                        f"{state['required_quantity']} "
                        f"{state['dish']} cooked successfully."
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # COOK FAILURE
    # --------------------------------------------------------

    print("COOK → FAILURE")

    remaining = state["cook_retry_attempts"] - 1

    return {
        "status": "cook_failed",

        "cook_retry_attempts": remaining,

        "messages": [
            AIMessage(
                content=(
                    "Sorry, there was a problem "
                    "while cooking your order."
                )
            )
        ]
    }


# ============================================================
# ROUTER AFTER COOK
# ============================================================

def route_after_cook(state: OrderState):

    # Cook succeeded
    if state["status"] == "ready":
        return "serve"

    # Cook failed but retry exists
    if state["cook_retry_attempts"] > 0:
        return "cook"

    # No cook retries left
    return "cook_failure"


# ============================================================
# NODE 6
# COOK FAILURE
# ============================================================

def cook_failure(state: OrderState):

    return {
        "status": "failed",

        "final_result": (
            "Order failed because cooking "
            "could not be completed."
        ),

        "messages": [
            AIMessage(
                content=(
                    "I am sorry, but I could not prepare "
                    "your order after multiple attempts."
                )
            )
        ]
    }


# ============================================================
# NODE 7
# SERVE
# ============================================================

def serve(state: OrderState):

    print("\n--------------------------------")
    print("SERVE")
    print("--------------------------------")

    print(
        "Remaining serve retries before this attempt:",
        state["serve_retry_attempts"]
    )

    # 60% SUCCESS
    # 40% FAILURE

    success = random.random() < 0.60

    # --------------------------------------------------------
    # SERVE SUCCESS
    # --------------------------------------------------------

    if success:

        print("SERVE → SUCCESS")

        return {
            "status": "complete",

            "final_result": (
                "Order completed successfully."
            ),

            "messages": [
                AIMessage(
                    content=(
                        f"Your order of "
                        f"{state['required_quantity']} "
                        f"{state['dish']} is complete!"
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # SERVE FAILURE
    # --------------------------------------------------------

    print("SERVE → FAILURE")

    remaining = state["serve_retry_attempts"] - 1

    return {
        "status": "serve_failed",

        "serve_retry_attempts": remaining,

        "messages": [
            AIMessage(
                content=(
                    "There was a problem while serving "
                    "your order."
                )
            )
        ]
    }


# ============================================================
# ROUTER AFTER SERVE
# ============================================================

def route_after_serve(state: OrderState):

    # --------------------------------------------------------
    # Serve successful
    # --------------------------------------------------------

    if state["status"] == "complete":
        return "end"

    # --------------------------------------------------------
    # Serve failed
    #
    # Requirement:
    #
    # Serve fails
    #     ↓
    # Cook again
    #     ↓
    # Serve again
    #
    # BUT cook retry must be available.
    # --------------------------------------------------------

    if state["status"] == "serve_failed":

        # Cook retry available
        if state["cook_retry_attempts"] > 0:

            # Serve retry must ALSO be available
            if state["serve_retry_attempts"] > 0:
                return "cook"

        return "serve_failure"

    return "serve_failure"


# ============================================================
# NODE 8
# SERVE FAILURE
# ============================================================

def serve_failure(state: OrderState):

    return {
        "status": "failed",

        "final_result": (
            "Order failed because serving "
            "could not be completed."
        ),

        "messages": [
            AIMessage(
                content=(
                    "I am sorry, but I could not complete "
                    "your order after multiple attempts."
                )
            )
        ]
    }


# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(OrderState)


# ============================================================
# ADD NODES
# ============================================================

builder.add_node(
    "extract_order",
    extract_order
)

builder.add_node(
    "order_confirm",
    order_confirm
)

builder.add_node(
    "order_decision",
    order_decision
)

builder.add_node(
    "new_order_input",
    new_order_input
)

builder.add_node(
    "cook",
    cook
)

builder.add_node(
    "cook_failure",
    cook_failure
)

builder.add_node(
    "serve",
    serve
)

builder.add_node(
    "serve_failure",
    serve_failure
)


# ============================================================
# START
# ============================================================

builder.add_edge(
    START,
    "extract_order"
)


# ============================================================
# EXTRACT ORDER
# ============================================================

builder.add_conditional_edges(
    "extract_order",

    route_after_extract,

    {
        "order_confirm": "order_confirm",
        "end": END
    }
)


# ============================================================
# ORDER CONFIRM
# ============================================================

builder.add_conditional_edges(
    "order_confirm",

    route_after_order_confirm,

    {
        "cook": "cook",
        "order_decision": "order_decision",
        "end": END
    }
)


# ============================================================
# ORDER DECISION
# ============================================================

builder.add_conditional_edges(
    "order_decision",

    route_after_order_decision,

    {
        "cook": "cook",
        "new_order_input": "new_order_input",
        "end": END
    }
)


# ============================================================
# NEW ORDER
# ============================================================

builder.add_edge(
    "new_order_input",
    "extract_order"
)


# ============================================================
# COOK
# ============================================================

builder.add_conditional_edges(
    "cook",

    route_after_cook,

    {
        "cook": "cook",
        "serve": "serve",
        "cook_failure": "cook_failure"
    }
)


builder.add_edge(
    "cook_failure",
    END
)


# ============================================================
# SERVE
# ============================================================

builder.add_conditional_edges(
    "serve",

    route_after_serve,

    {
        "cook": "cook",
        "end": END,
        "serve_failure": "serve_failure"
    }
)


builder.add_edge(
    "serve_failure",
    END
)


# ============================================================
# COMPILE
# ============================================================

graph = builder.compile()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("     RESTAURANT ORDER AI AGENT")
    print("======================================")

    user_input = input(
        "\nWhat would you like to order? "
    ).strip()

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state: OrderState = {

        "messages": [
            HumanMessage(content=user_input)
        ],

        "dish": "",

        "required_quantity": 0,

        "available_quantity": "0",

        "status": "started",

        # 3 ORDER RETRIES
        "order_retry_attempts": 3,

        # INITIAL COOK + 2 RETRIES
        "cook_retry_attempts": 2,

        # INITIAL SERVE + 2 RETRIES
        "serve_retry_attempts": 2,

        "final_result": ""
    }

    # --------------------------------------------------------
    # EXECUTE GRAPH
    # --------------------------------------------------------

    result = graph.invoke(initial_state)

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n======================================")
    print("             FINAL RESULT")
    print("======================================")

    print(
        "\nStatus:",
        result["status"]
    )

    print(
        "Final result:",
        result["final_result"]
    )

    print(
        "\nDish:",
        result["dish"]
    )

    print(
        "Required quantity:",
        result["required_quantity"]
    )

    print(
        "Available quantity:",
        result["available_quantity"]
    )

    print(
        "\nRemaining order retries:",
        result["order_retry_attempts"]
    )

    print(
        "Remaining cook retries:",
        result["cook_retry_attempts"]
    )

    print(
        "Remaining serve retries:",
        result["serve_retry_attempts"]
    )