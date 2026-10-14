"""Eval suite. DeepEval reads its settings at import, so set them here, before
any evals.* module imports deepeval."""

import os

# A judge call on five 10-K excerpts can be slow under load, and one timed-out
# call fails the whole run. Each attempt gets 3 minutes, each metric / test case
# 10 minutes in total, with up to 3 attempts. (harness.JUDGE_ASYNC also caps how
# many judge calls run at once, which is what made calls slow in the first place.)
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "3")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
