"""Detector 3: LLM-as-judge via Groq.

Sends source and response to a general-purpose language model and asks it to
rule on whether the answer is supported. Unlike the other two it returns an
*explanation* — the specific claim it considers unsupported — which is what
makes the product's verdict readable rather than a bare number.

It is also the only detector that costs anything: every call spends quota
against a daily cap of 200,000 tokens. Caching lives in the harness, not
here, so that all three detectors are cached the same way.
"""

import json      # the judge is asked to reply in JSON; this parses it
import os        # reads GROQ_API_KEY out of the environment
import time      # latency measurement, same as the other two detectors

from dotenv import load_dotenv  # functiona from python_dotenv library, reads .env into the environment
from gradio.monitoring_dashboard import data
from groq import Groq

from src.base import DetectorResult
from src.halueval import TestCase

# Kept as a constant so a model change is a one-line edit. That is not
# hypothetical: Llama 3.3 was retired from Groq's free tier mid-project and
# had to be swapped out for this one.
MODEL_NAME = "openai/gpt-oss-120b"

# Two things this prompt has to get right:
#
# 1. "Not mentioned" must count as hallucinated. Left unsaid, a model tends to
#    flag only direct contradictions — and invented detail that the source
#    simply never covers is the more common kind of hallucination.
# 2. Naming the offending claim is both a product requirement and an accuracy
#    device: a model obliged to point at something specific guesses less.
#
# JSON is requested so the reply can be parsed structurally instead of having
# meaning extracted from prose with regular expressions.
SYSTEM_PROMPT = """You decide whether an AI-generated answer is supported by a source text.

Rules:
- "grounded": every claim in the answer is supported by the source.
- "hallucinated": the answer contains at least one claim the source does not
  support. A claim the source simply does not mention counts as hallucinated,
  not only a claim the source contradicts.
When the verdict is "hallucinated", also say which kind it is:
- "contradicted": the source states something incompatible with the claim.
- "not_mentioned": the source neither confirms nor denies the claim.

Reply with JSON only, no other text, in exactly this shape:
{"verdict": "grounded" or "hallucinated", "unsupported": "the specific unsupported claim, or an empty string when grounded", "kind": "contradicted" or "not_mentioned" or ""}"""

def _build_user_message(case: TestCase) -> str:
    """Assemble the text the judge reads.

    Leading underscore: internal to this module, not part of its interface.
    """
    # Labelled blocks rather than raw text runs: without markers the model has
    # to guess where the source ends and the answer begins.
    parts = [f"SOURCE:\n{case.source}"]

    # Only the QA subset carries a question. Including it matters more than it
    # looks: without it, a short answer such as "Delhi" is a bare word rather
    # than a statement, and there is nothing for the judge to verify.
    if case.context:
        parts.append(f"QUESTION ASKED:\n{case.context}")

    parts.append(f"AI ANSWER:\n{case.response}")

    # Blank line between blocks — join inserts the separator *between* items,
    # not after each, so no trailing newline is added.
    return "\n\n".join(parts)


def _parse(raw: str) -> tuple[str, str, str]:
    """Pull verdict, explanation and kind out of the model's reply."""
    # Written defensively rather than calling json.loads(raw) directly: a model
    # asked for JSON still sometimes wraps it in markdown fences or adds a
    # sentence around it. Taking everything between the outermost braces
    # survives both.
    start = raw.find("{")   # position of the first opening brace
    end = raw.rfind("}")    # rfind searches backwards — the last closing brace

    if start == -1 or end == -1:
        # find() returns -1 when nothing matches. No braces at all means the
        # reply is not JSON in any form, and there is nothing to salvage.
        raise ValueError(f"no JSON in judge reply: {raw[:200]}")

    # end + 1 because the right edge of a slice is excluded, and the closing
    # brace has to be inside the parsed text.
    data = json.loads(raw[start : end + 1])

    verdict = data.get("verdict")

    # Raise rather than defaulting to a verdict. A silent substitution would
    # put rulings into the results that the model never made, and the metrics
    # computed from them would be quietly wrong. A visible failure is cheaper.
    if verdict not in ("grounded", "hallucinated"):
        raise ValueError(f"unexpected verdict in judge reply: {verdict!r}")

    # !r above shows the value the way it appears in code, with quotes — which
    # distinguishes an empty string from a missing value in the message.

    # Empty string when grounded: there is no unsupported claim to name.
    kind = data.get("kind", "")
    # Validated more loosely than the verdict on purpose: this field is
    # auxiliary, so an unexpected value should not abort a run that costs
    # quota. Anything unrecognised becomes empty rather than polluting the
    # analysis with a category the prompt never defined.
    if kind not in ("contradicted", "not_mentioned"):
        kind = ""
    # The JSON calls this "unsupported" because that tells the model exactly
    # what to put there; DetectorResult calls it "explanation" because that
    # slot is common to all detectors. Same value, two names.
    return verdict, data.get("unsupported", ""), data.get("kind", "")


class LLMJudgeDetector:
    # Required by the Detector protocol in base.py.
    name = "llm_judge"

    # Bump this whenever SYSTEM_PROMPT changes. The version is part of the
    # cache key, so without a bump the store would serve verdicts produced by
    # the old prompt under the new one — meaning a prompt edit has no effect
    # at all, silently.
    # v2: added the "kind" field to the prompt. The bump matters — without it
    # the store would serve v1 verdicts, which carry no kind at all.
    version = "v2"

    def __init__(self):
        # Reads .env into the process environment. Cheap enough to do here,
        # unlike the model loading in detectors 1 and 2.
        load_dotenv()

        # Subscript rather than os.environ.get(): a missing key should fail
        # immediately and obviously, not surface later as a confusing error
        # from the server about an empty credential.
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def check(self, case: TestCase) -> DetectorResult:
        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            # Chat models take a list of messages. The system message carries
            # the instructions, the user message carries the material — keeping
            # them apart stops the data from reading as further instructions.
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(case)},
            ],
            # Temperature controls randomness; zero means always take the most
            # likely continuation. Required here: without it the same input can
            # yield different verdicts, and the comparison would be measuring
            # the model's noise instead of the difference between methods.
            temperature=0,
        )

        # Measured before parsing, so the number reflects the API round trip
        # and not our own processing.
        latency_ms = (time.perf_counter() - start) * 1000

        # choices is a list because the API can return several alternatives;
        # we asked for one, so the first is the only one.
        content = response.choices[0].message.content

        # The API may return an empty message — a content filter, or an
        # abnormal finish. There is nothing to parse, and passing None along
        # would surface later as a confusing error far from its cause.
        if content is None:
            raise ValueError("judge returned no content")

        verdict, explanation, kind = _parse(content)

        return DetectorResult(
            verdict=verdict,
            # The judge rules, it does not measure — there is no natural score
            # to report, so it is derived from the verdict. This is precisely
            # why this detector has no tunable threshold, unlike the other two.
            
            score=1.0 if verdict == "grounded" else 0.0,
            latency_ms=latency_ms,
            explanation=explanation,
            unsupported_kind=kind,
            # Includes the hidden reasoning tokens: check_groq.py showed 38 of
            # them spent on a one-word answer, so real calls carry that on top
            # of the visible reply.
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )



# check
if __name__ == "__main__":
    from src.halueval import build_splits

    # The underscore means "second return value not needed here".
    dev, _ = build_splits()

    detector = LLMJudgeDetector()

    

    # Three QA cases is enough to see the pattern without waiting.
    for case in dev[:3]:
        result = detector.check(case)
        print(f"{case.case_id}")
        print(f"  expected: {case.label}")
        print(f"  verdict:  {result.verdict}")
        print(f"  says:     {result.explanation[:100]}")
        print(f"  kind:     {result.unsupported_kind or '—'}")
        print(f"  tokens:   {result.tokens_used}, {result.latency_ms:.0f} ms")
        print()
        