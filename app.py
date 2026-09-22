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
#
# Thresholds are stated explicitly and must match run_eval.py, or the app would
# give verdicts at a threshold nothing was ever measured at. HHEM ships at the
# dev-tuned 0.46. The baseline ships at the naive 0.50 — its own tuning chose
# 0.00, which calls everything grounded: honest as a measurement, useless as a
# detector.
embeddings = EmbeddingDetector(threshold=0.50)
entailment = EntailmentDetector(threshold=0.46)
judge = LLMJudgeDetector()


# One wording per verdict. The judge also reports whether an unsupported claim
# contradicts the source or is simply absent from it, but that distinction is
# not stable: identical input comes back as either one on different runs. So
# both look the same on screen, and the fallback below uses this dict too — the
# same outcome reads the same whichever method produced it.
VERDICT_WORDING = {
    "grounded": "✅ **Supported by your source**",
    "hallucinated": "⚠️ **Not supported by your source**",
}

# Measured on the 300-case test split, all three methods, zero failures.
ACCURACY = {"embeddings": 0.357, "entailment": 0.640, "judge": 0.817}

# "grounded" and "hallucinated" are our internal words. On screen a visitor
# reads plain ones, and the same two words are used for every method so the
# rows of the breakdown can be compared at a glance.
OUTCOME = {"grounded": "no error found", "hallucinated": "error found"}


def breakdown_table(embedding=None, entailment_result=None, judge=None,
                    judge_unavailable=False) -> str:
    """The three-method table, with a dash wherever a check has not run.

    The first two columns hold before anything is checked — what each method is
    and how often it is right in general — so the table doubles as the panel's
    empty state. A visitor reads what the product does before pressing anything,
    and no space sits blank to no purpose.
    """
    dash = "—"

    def cells(result):
        if result is None:
            return dash, dash, dash
        return (OUTCOME[result.verdict], f"{result.score:.2f}",
                f"{result.latency_ms:.0f} ms")

    sim_v, sim_s, sim_t = cells(embedding)
    ent_v, ent_s, ent_t = cells(entailment_result)
    # The judge rules rather than scores, so its score cell is always a dash.
    # "unavailable" and "not run yet" must not look alike: one is a failure the
    # visitor should know about, the other is simply the starting state.
    if judge is not None:
        judge_v, judge_t = OUTCOME[judge.verdict], f"{judge.latency_ms:.0f} ms"
    elif judge_unavailable:
        judge_v, judge_t = "unavailable", dash
    else:
        judge_v, judge_t = dash, dash

    return "\n".join([
        "| Method | What it does | Verdict | Score | Speed |",
        "|---|---|---|---|---|",
        f"| Text similarity | Compares overall wording, not facts. Right on "
        f"{ACCURACY['embeddings']:.0%} of 300 test answers. | {sim_v} | {sim_s} | {sim_t} |",
        f"| Entailment | Asks whether the answer follows from the document. Right "
        f"on {ACCURACY['entailment']:.0%}. | {ent_v} | {ent_s} | {ent_t} |",
        f"| AI judge | Reads both and names the unsupported claim. Right on "
        f"{ACCURACY['judge']:.0%}. | {judge_v} | {dash} | {judge_t} |",
        "",
        "*Each score means something different: for text similarity it is how "
        "close the two texts are in wording (0 to 1), for entailment it is how "
        "confident the model is that the answer follows from the document (0 to "
        "1). The AI judge gives a ruling rather than a number, so it has no "
        "score. The percentages are how often each method agreed with a human on "
        "the same 300 answers from HaluEval — a public research set in which "
        "people marked which AI answers were faithful to their source — and only "
        "those are comparable between methods.*",
    ])


def check(source: str, response: str) -> tuple[str, str]:
    """Run all three detectors and phrase the outcome for a reader.

    Returns the verdict a visitor reads, the source with the relevant passage
    marked, and the method-by-method breakdown behind it.
    """
    # Run on two empty strings the detectors agree enthusiastically — the cosine
    # similarity of nothing with nothing is 1.00 — and the screen then asserted
    # that an empty answer was supported by an empty document. Checking first
    # also saves a judge call, which costs quota.
    if not source.strip() or not response.strip():
        return ("*Nothing to check yet — paste a document and an answer first.*",
                breakdown_table())

    # TestCase was designed for evaluation, where the correct answer is known.
    # Here it is not — that is the entire question the user is asking — so the
    # label is empty and the id is a placeholder.
    case = TestCase(
        case_id="live", # The id is a placeholder, not a real case. 
        # The label is empty because the user is asking the question, 
        # so the correct answer is not known.
        subset="live", # The subset is a placeholder, not a real subset.
        source=source, # The source is the document the user provided.
        response=response, # The response is the answer the user provided.
        label="", # The label is empty because the user is asking the question,
        # so the correct answer is not known.
    )

    # These two are local, free and unlimited detectors, so they always run.
    embedding_result = embeddings.check(case)
    entailment_result = entailment.check(case)

    # The judge is the only detector that can become unavailable: it calls an
    # outside service on a free tier that runs out. Its failure must reach the
    # screen as a sentence — the other two still have something to say.
    judge_result = None
    try:
        judge_result = judge.check(case)
    except Exception as error:
        # Logged rather than displayed: the visitor needs the sentence below,
        # the cause belongs in the Space's log.
        print(f"judge failed: {error}")

    lines = []  # Joined into the verdict block below.

    if judge_result is not None:
        # The kind (contradicted or not_mentioned) is deliberately not shown: the
        # same input returns either one on different runs, so wording or styling
        # built on it would change between two clicks.
        lines.append(VERDICT_WORDING[judge_result.verdict])

        if judge_result.explanation:
            # Never styled as a quotation: the checking model usually paraphrases
            # the claim rather than repeating it, and passing a paraphrase off as
            # a quote would be the product doing the very thing it exists to catch.
            lines.append(f"\n**What the AI flagged:** {judge_result.explanation}")

        # Disagreement is a feature, not an edge case: when the two methods that
        # carry signal split, that split is the best available sign the verdict
        # is uncertain. The baseline is excluded on purpose — at 0.357 it
        # disagrees constantly and means nothing by it.
        if judge_result.verdict != entailment_result.verdict:
            # Two different situations, and the wording must not blur them. When
            # the judge flagged something, that claim is the thing to look at.
            # When it did not, nothing has been named — and promising the reader
            # a "part" to re-read would be inventing one.
            if judge_result.explanation:
                lines.append(
                    "\n*The two methods disagree: the other check reads this answer "
                    "as supported. The claim named above is the one in question — "
                    "check it against your document yourself.*"
                )
            else:
                lines.append(
                    "\n*The two methods disagree: the AI found nothing unsupported, "
                    "while the other check was not convinced. Nothing specific was "
                    "named — worth reading the answer against your document yourself.*"
                )
    else:
        # Fallback: HHEM decides, and the loss is stated rather than hidden —
        # it returns a number, so no explanation of any kind is available.
        lines.append(VERDICT_WORDING[entailment_result.verdict])
        lines.append(
            "\n*The AI check is unavailable — the free daily allowance is used up "
            "and resets at midnight UTC. This verdict comes from a smaller model "
            "running here, which gives an answer but cannot say which part is "
            "unsupported.*"
        )

    # One row per method, always on screen rather than hidden behind a disclosure:
    # comparing the three is the point of the project, and a collapsed panel also
    # left a hole in the layout. The columns answer three different questions —
    # what the method is, what it said about these two texts, and how often it is
    # right in general. The last must not read as a property of this check, which
    # is why it has its own column and the footnote below.
    return "\n".join(lines), breakdown_table(
        embedding_result, entailment_result, judge_result,
        judge_unavailable=judge_result is None,
    )


# gr.Blocks: everything created inside the `with` is attached to the page, in
# the order it appears. Nothing is wired automatically — the button below has
# to be connected by hand, which is exactly the control this UI needs.
# text_sm makes every label, field and button one step smaller. A theme is the
# supported lever for this; CSS overrides tend to break when Gradio changes its
# internals, and this file has to survive on the Space untouched.
CSS = """
/* Gradio's hint grey is #bbbbc2 — about 1.9:1 against white, well under the
   4.5:1 WCAG AA asks for. Redefining the variable moves every hint, the caption
   and the breakdown to the grey Gradio already uses for field labels (4.8:1),
   so they match each other and stay readable. */
.gradio-container {
  --block-info-text-color: #565e6b;
  --block-title-text-color: #565e6b;
}

#when-line .md.prose,
#result-caption .md.prose,
#breakdown-panel .md.prose,
#breakdown-panel table {
  font-size: var(--block-info-text-size);
  line-height: var(--line-sm);
}
#when-line .md.prose *,
#result-caption .md.prose *,
#breakdown-panel .md.prose * { color: var(--block-info-text-color); }

/* A tinted page makes the two panels read as cards sitting on it, rather than
   as text floating on the same white as everything else. */
.gradio-container { background: #f1f2f4; }

/* Gradio centres the page inside a narrower container, leaving about 120px
   unused down each side. Taking that back widens both columns, which lets a
   pasted article fit in fewer lines — the reason the examples had fallen off
   the bottom of the screen.
   Two caps, not one: the container, and `main.fillable` inside it, which has a
   max-width of its own at 1280px and is what actually held the columns to 600.
   Two caps hold the columns at 600: the container's own max-width, and a second
   one on <main> — written inline by Gradio, which is why the rule below needs
   !important. Widening both takes each column to 680, and a pasted article then
   fits in nine lines instead of overflowing them. */
body /* Centred, not flush left: capped at 1440 in a wider window it left 32px of
   margin on one side and 62px on the other. */
.gradio-container { max-width: 1440px; margin-inline: auto; }

/* The container is capped and sits flush left, so on a wider window its right
   edge left an uncovered strip showing white through. The tint has to sit on
   the document itself. `gradio-app` needs !important for the same reason the
   width rule does: Gradio writes its background as an inline style, and no
   stylesheet rule outranks one. */
body { background: #f1f2f4; }
gradio-app { background: #f1f2f4 !important; }
/* !important is not decoration here: Gradio writes this cap as an inline style
   on <main>, and no stylesheet rule outranks an inline style without it. */
body .gradio-container main.app.fillable { max-width: 1440px !important; }

/* The right column is one card, like the bordered form on the left. Gradio
   paints a group's inner .styler layer with the border colour, so that the
   hairlines between stacked fields show through; on the left that grey is
   hidden by the white blocks of the fields themselves, but here the children
   are transparent Markdown and it showed as a grey panel. */
#result-card, #result-card .styler { background: var(--block-background-fill); }

/* The card needs its own side padding. On the left the inset comes from each
   field's own block; the Markdown blocks here carry `hide-container`, which
   strips that padding, so every line sat 1px from the border. 12px matches the
   13px the left column insets its label and textarea by. Sides only: top
   padding would push the label down and undo the alignment below. */
#result-card .styler { padding-left: 12px; padding-right: 12px; }

/* "Audit verdict" sits on the same line as "Your document", and the caption on
   the same line as that field's hint. Measured: the card's first block starts
   10px higher than the form's first label. */
#result-label { margin-top: 10px; }
#result-label .md.prose * {
  color: var(--block-title-text-color);
  font-size: var(--block-title-text-size);
  font-weight: var(--block-title-text-weight);
}
#result-caption { margin-top: -1px; }

/* The panel keeps its size whatever it holds: a result that resized the page
   moved the button and the examples under the reader's cursor. Taller content
   scrolls inside instead. Height is the left column measured from the label to
   the bottom of the button. */
/* The numeric columns are monospaced with tabular figures, so every digit sits
   in the same width and the column reads down as a list of measurements. */
#breakdown-panel td:nth-child(n+4) {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

/* The fields take the same size and colour as the hint above them. They hold
   text rather than present it, and at near-black they were the heaviest thing
   on the page — which put the weight on the input rather than on the verdict.
   The 1.6 leading stays: it is what makes a pasted wall of text legible at all.
   Addressed by id because Gradio styles its own textareas through a class of
   equal weight, and its stylesheet loads last, so an equally specific selector
   of ours silently loses. */
#source-field textarea, #answer-field textarea {
  font-size: var(--block-info-text-size);
  line-height: 1.6;
  color: var(--block-info-text-color);
}

/* The card behind the labels takes the page's colour, so the left column stops
   reading as the main object on screen — the value is the verdict, and it was
   losing to two white blocks of near-black text. The fields themselves stay
   white: an input that does not look like an input stops inviting a paste. */
#input-card .form,
#input-card .block { background: #f1f2f4; }

/* Numbers must not wrap: "Score" broke across two lines, and so did "2051 ms". */
#breakdown-panel th:nth-child(1), #breakdown-panel td:nth-child(1),
#breakdown-panel th:nth-child(n+3), #breakdown-panel td:nth-child(n+3) {
  white-space: nowrap;
}

#result-body {
  height: 389px;
  overflow-y: auto;
  display: block;
}

/* The verdict is the one thing the visitor came for, so it is the largest text
   in the panel. Scoped to bold in the first paragraph: the label on the flagged
   claim is also bold but must stay ordinary size. */
#verdict-panel .md.prose p:first-child strong { font-size: 1.3rem; }

/* Everything italic in the panel is an aside — the note about disagreement,
   the unavailable-judge message, the footnote under the table. */
#verdict-panel .md.prose em { color: var(--block-info-text-color); }
"""

with gr.Blocks(
    title="AI Output Auditor",
    # primary_hue sets the colour of the primary button. The default orange
    # shouts on a page whose whole subject is calibrated confidence; slate
    # still reads as the one action to take, without the alarm.
    # Inter is drawn for screens at small sizes, and this page is mostly small
    # text: two field hints, a caption and a footnote at 11px. A monospace face
    # carries the numeric columns so digits line up down the table.
    theme=gr_themes.Default(
        text_size=gr_themes.sizes.text_sm,
        primary_hue=gr_themes.colors.slate,
        font=gr_themes.GoogleFont("Inter"),
        font_mono=gr_themes.GoogleFont("IBM Plex Mono"),
    ),
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



    # A Row places its children side by side; a Column stacks them. equal_height
    # is off on purpose: it forces both columns to the same height and hands the
    # spare space to every child, which stretched the button whenever the result
    # grew and left a gap above the breakdown when it did not.
    with gr.Row(equal_height=False):
        with gr.Column(elem_id="input-card"):
            # autoscroll=False keeps a pasted document showing its first line:
            # Gradio otherwise scrolls a textbox to the end on every change, so
            # an example arrived mid-article with its opening hidden.
            # max_lines matches lines on purpose. Without it a Gradio textbox
            # grows with its content: pasting an article pushed the button and
            # the examples down the page, and left the fixed result card no
            # longer level with them. Fixed height, scroll inside.
            source_box = gr.Textbox(
                label="Your document",
                info="Paste the full text of your own document — an article, a "
                     "contract, a report. Not something the AI wrote for you.",
                lines=9, max_lines=9, autoscroll=False,
                elem_id="source-field",
            )
            response_box = gr.Textbox(
                label="The AI's answer",
                info="Paste only what the AI replied. Your question isn't needed — "
                     "the answer is checked against the document above.",
                lines=4, max_lines=4, autoscroll=False,
                elem_id="answer-field",
            )
            check_button = gr.Button("Check the answer", variant="primary")

        with gr.Column():
          with gr.Group(elem_id="result-card"):
            gr.Markdown("Audit verdict", elem_id="result-label")
            # Static: true of every check, so it does not wait for one.
            gr.Markdown(
                "Three methods run on every check. The headline comes from the AI "
                "judge — the only one that can name what is unsupported.",
                elem_id="result-caption",
            )
            # Markdown rather than a Textbox: the verdict is prose to be read,
            # not a value to be edited.
            with gr.Column(elem_id="result-body"):
                verdict_box = gr.Markdown(elem_id="verdict-panel")
                breakdown_box = gr.Markdown(breakdown_table(), elem_id="breakdown-panel")


    # The wiring Interface used to do for us: on click, call check() with the
    # contents of these two boxes and spread its two return values across the
    # verdict and the breakdown. Outputs are matched by position, not by name.
    check_button.click(
        fn=check,
        inputs=[source_box, response_box],
        outputs=[verdict_box, breakdown_box],
    )

    # A result is only true of the inputs it was computed from. The moment either
    # one changes — including when clicking an example fills them — the panel is
    # cleared, so the screen can never show a new document beside an old verdict.
    # This is why the button stays: running on every keystroke would spend quota
    # on half-pasted text and give the visitor no say in sending their document
    # to an outside service.
    for box in (source_box, response_box):
        box.change(fn=lambda: ("", breakdown_table()),
                   outputs=[verdict_box, breakdown_box])


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
