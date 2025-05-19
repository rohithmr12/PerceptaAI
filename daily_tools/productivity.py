# daily_tools/productivity.py
from typing import List, Dict, Union

# In-memory storage for the to-do list, managed by this module
_todo_list_storage: List[str] = []

def add_todo_item(item: str) -> Dict[str, Union[str, int]]:
    """Adds an item to the to-do list."""
    if not item.strip():
        # This check could also be in the server endpoint, but good to have here too.
        raise ValueError("To-do item cannot be empty.") 
    _todo_list_storage.append(item)
    return {"message": f"'{item}' added.", "total_items": len(_todo_list_storage)}

def view_todo_list() -> List[str]:
    """Views all items in the to-do list."""
    if not _todo_list_storage:
        return ["To-do list is empty."]
    return [f"{i+1}. {item}" for i, item in enumerate(_todo_list_storage)]

def remove_todo_item(item_number: int) -> str:
    """Removes an item from the to-do list by its 1-based index."""
    idx = item_number - 1  # Convert to 0-based index
    if not (0 <= idx < len(_todo_list_storage)):
        raise IndexError(f"Invalid item number. Use 1 to {len(_todo_list_storage)}.")
    removed_item = _todo_list_storage.pop(idx)
    return f"Removed '{removed_item}'."

def calculate_expression(expression: str) -> float:
    """Calculates a simple arithmetic expression. 
       WARNING: Uses eval(), which is a security risk with untrusted input.
       The server should ensure the expression is sanitized or use a safer parsing method.
    """
    # Basic validation (server should ideally do more robust sanitization)
    allowed_chars = "0123456789+-*/(). " 
    if not all(char in allowed_chars for char in expression) or not expression.strip():
        raise ValueError("Invalid characters or empty expression provided.")
    try:
        # Eval is dangerous with unsanitized input. Consider a safer math expression parser for production.
        result = eval(expression)
        return float(result)
    except ZeroDivisionError:
        raise ZeroDivisionError("Error: Division by zero.")
    except Exception as e:
        # Catch other eval-related errors (SyntaxError, NameError etc.)
        raise ValueError(f"Invalid expression: {str(e)}") 