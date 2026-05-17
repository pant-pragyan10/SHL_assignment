from shl_agent.eval.evaluator import Evaluator


def test_evaluator_smoke():
    ev = Evaluator(retriever=None, orchestrator=None, catalog=[{"name":"A","url":"http://x","tags":["python"]}])
    # schema validation
    ok, msg = ev.validate_schema({"reply":"hi","recommendations":[],"end_of_conversation":False})
    assert ok
