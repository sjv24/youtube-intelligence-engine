import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def final_agent(state):
    # pick whichever answer was populated
    answer = (
        state.get("final_answer") or
        state.get("script_answer") or
        state.get("textbook_answer") or
        state.get("comments_answer") or
        "No answer generated."
    )
    return {"final_answer": answer}