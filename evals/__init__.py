"""Eval suite. DeepEval reads its settings at import, so set them here, before
any evals.* module imports deepeval."""

import os

# A judge call on five 10-K excerpts can be slow under load; DeepEval's default
# per-task budget times out a whole run over one slow call. Give each metric /
# test case 5 minutes including retries, and retry transient API errors.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "3")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
