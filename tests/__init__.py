import logging

# Expected upstream-failure warnings are noise in test output. A NullHandler
# keeps them off stderr while `assertLogs` can still capture them.
logging.getLogger("llm_gateway").addHandler(logging.NullHandler())
