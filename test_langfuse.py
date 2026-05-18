import os
from dotenv import load_dotenv
load_dotenv()
from langfuse import Langfuse

client = Langfuse()
trace = client.trace(name="test_trace_from_script", input="hello")
trace.generation(name="test_gen", model="gpt-3.5", input="hello", output="world")
trace.score(name="test_score", value=1.0)
client.flush()
print("Flushed trace:", trace.id)
