"""AI Output Auditor — all three detectors behind a Gradio interface."""


import gradio as gr
# `spaces` is a HuggingFace package that exists only on a Space. Locally there
# is no GPU to reserve, so a no-op stands in and the same file runs in both
# places — which is the point: what you test here is what deploys there.
try:
    from spaces import GPU  # type: ignore
except ImportError:
    def GPU(**kwargs):
        return lambda fn: fn

from src.embeddings import EmbeddingDetector
from src.entailment import EntailmentDetector
from src.halueval import TestCase
from src.llm_judge import LLMJudgeDetector
import gradio.themes as gr_themes
from examples import EXAMPLES, EXAMPLE_LABELS


# A ZeroGPU Space refuses to start without at least one @spaces.GPU function:
# the first attempt failed with "No @spaces.GPU function detected during
# startup". Nothing here needs a GPU — MiniLM is 22M parameters, HHEM 110M,
# both comfortable on CPU, and the judge is an HTTP call to Groq.
#
# So the decorator sits on a function that satisfies the requirement without
# taking part in the work. Decorating the real check instead would spend the
# 5-minute daily GPU quota and add a queue wait before every request, in
# exchange for no speed-up at all.
@GPU(duration=10)
def _zerogpu_requirement() -> None:
    """Not called. Present so the platform allows the Space to start."""
    return None


# Built once at import time rather than inside check(). About 530 MB of weights
# download on a cold start, and doing it here puts that wait on the Space's own
# startup instead of on whoever opens the page first. It also means a broken
# model shows up as a failed startup, which is far easier to diagnose than a
# request that mysteriously errors.
embeddings = EmbeddingDetector()
entailment = EntailmentDetector()
judge = LLMJudgeDetector()


def check(source: str, response: str) -> str:
    """Run all three detectors on one source/response pair."""
    # TestCase was designed for evaluation, where the correct answer is known.
    # Here it is not — that is the entire question the user is asking — so the
    # label is empty and the id is a placeholder. Worth revisiting later: a
    # product should not have to invent evaluation metadata to ask a detector
    # a question.
    case = TestCase(
        case_id="live",
        subset="live",
        source=source,
        response=response,
        label="",
    )

    lines = []
    for name, detector in (("Embeddings", embeddings), ("HHEM", entailment)):
        result = detector.check(case)
        lines.append(
            f"{name}: {result.verdict}  "
            f"(score {result.score:.4f}, {result.latency_ms:.0f} ms)"
        )

    # The judge is the only detector that can become unavailable: it calls an
    # outside service on a free tier that runs out. Its failure has to reach
    # the screen as a sentence — the other two still have something useful to
    # say, and a visitor must not be shown a stack trace or an empty panel.
    try:
        result = judge.check(case)
        lines.append(
            f"Judge: {result.verdict}  "
            f"({result.latency_ms:.0f} ms, {result.tokens_used} tokens)"
        )
        if result.explanation:
            lines.append(f"unsupported: {result.explanation}")
    except Exception as error:
        lines.append(
            "Judge: unavailable. The free-tier limit may be used up — "
            "it resets at midnight UTC."
        )
        # Logged rather than displayed: the visitor needs the sentence above,
        # the actual cause belongs in the Space's log.
        print(f"judge failed: {error}")

        
    return "\n".join(lines)
    


# gr.Blocks: everything created inside the `with` is attached to the page, in
# the order it appears. Nothing is wired automatically — the button below has
# to be connected by hand, which is exactly the control this UI needs.
# text_sm makes every label, field and button one step smaller. A theme is the
# supported lever for this; CSS overrides tend to break when Gradio changes its
# internals, and this file has to survive on the Space untouched.
# The result panel stretches with the row, but the textarea inside keeps the
# height its `rows` attribute gives it. Gradio's own height:100% rule is scoped
# to labels without a container, so a bordered, labelled field never fills its
# panel. The first three rules restore that chain for the result field; the last
# two give the "when is this worth doing" line the same grey as every field hint,
# using the variables Gradio's own Info component uses.
CSS = """
#result-panel label.container { display: flex; flex-direction: column; height: 100%; }
#result-panel .input-container { flex: 1 1 auto; align-items: stretch; }
#result-panel textarea { height: 100%; }

#when-line .md.prose {
  font-weight: var(--block-info-text-weight);
  font-size: var(--block-info-text-size);
  line-height: var(--line-sm);
}
#when-line .md.prose * { color: var(--block-info-text-color); }
"""

with gr.Blocks(
    title="AI Output Auditor",
    theme=gr_themes.Default(text_size=gr_themes.sizes.text_sm),
    css=CSS,
) as demo:
    gr.Markdown("# AI Output Auditor")
    gr.Markdown(
        "Gave an AI a document and asked it something — and want to be sure of the "
        "answer? Paste both below. This checks the answer against your document and "
        "tells you either that it holds up, or exactly which parts your document "
        "does not back up."
    )
    gr.Markdown(
        "Most worth doing when you asked for figures, dates or names, when the answer "
        "may go beyond what your document covers, or when the document is long.",
        elem_id="when-line",
    )



    # A Row places its children side by side; a Column stacks them. Nesting
    # columns inside a row gives the two-panel layout: inputs on the left,
    # result on the right. equal_height makes the two columns end level.
    with gr.Row(equal_height=True):
        with gr.Column():
            source_box = gr.Textbox(
                label="Your document",
                info="Paste the full text of your own document — an article, a "
                     "contract, a report. Not something the AI wrote for you.",
                lines=9,
            )
            response_box = gr.Textbox(
                label="The AI's answer",
                info="Paste only what the AI replied. Your question isn't needed — "
                     "the answer is checked against the document above.",
                lines=4,
            )
            check_button = gr.Button("Check the answer")

        with gr.Column():
            result_box = gr.Textbox(
                label="Result",
                info="Whether your document backs the answer. Three methods run, and "
                     "the judge — itself an AI, so a second opinion rather than a "
                     "verdict — names any part that is not supported.",
                lines=6, max_lines=6, elem_id="result-panel",
            )


    # The wiring Interface used to do for us: on click, call check() with the
    # contents of these two boxes and put the return value into that one.
    check_button.click(
        fn=check,
        inputs=[source_box, response_box],
        outputs=result_box,
    )

    # Examples must be created explicitly now, and told which components they
    # fill. Both rows are real cases from the reality check; the texts, their
    # Wikipedia attribution and the labels live in examples.py.

    gr.Examples(
        examples=EXAMPLES,
        inputs=[source_box, response_box],
        example_labels=EXAMPLE_LABELS,
    )



# SSR is on by default and shuts the app down immediately on Spaces — a known
# Gradio issue, unrelated to ZeroGPU.
demo.launch(ssr_mode=False)
