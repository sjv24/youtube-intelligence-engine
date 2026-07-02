import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from agents.script_agent import script_agent
from agents.textbook_agent import textbook_agent
from agents.comments_agent import comments_agent

lm = dspy.LM('ollama/qwen3:8b',
             api_base='http://localhost:11434',
             api_key=None,
             temperature=0.9)

class synthesize(dspy.Signature):
    """synthesize answers from script, textbook and fan comments into one complete answer"""
    question = dspy.InputField(desc="original question")
    script_answer = dspy.InputField(desc="answer from movie script")
    textbook_answer = dspy.InputField(desc="answer from science textbook")
    comments_answer = dspy.InputField(desc="what fans say about this")
    final_answer = dspy.OutputField(desc="synthesized answer combining all three perspectives")

gen_synthesize = dspy.Predict(synthesize)

def multi_agent(state):
    question = state["question"]
    
    script_result = script_agent(state)
    textbook_result = textbook_agent(state)
    comments_result = comments_agent(state)

    with dspy.context(lm=lm):
        result = gen_synthesize(
            question=question,
            script_answer=script_result.get("script_answer", ""),
            textbook_answer=textbook_result.get("textbook_answer", ""),
            comments_answer=comments_result.get("comments_answer", "")
        )

    return {"final_answer": result.final_answer}