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
from groq import RateLimitError
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
# Built inside a try for one reason: __init__ reads GROQ_API_KEY with a
# subscript, so a missing secret raises here — at import, before the interface
# exists — and takes the whole page down, including the two detectors that need
# no key at all. Built this way, a missing key costs the judge alone, and
# check() already knows what to do when the judge is not there.
try:
    judge = LLMJudgeDetector()
except Exception as error:
    print(f"judge unavailable at startup: {error}")
    judge = None


# One wording per verdict. The judge also reports whether an unsupported claim
# contradicts the source or is simply absent from it, but that distinction is
# not stable: identical input comes back as either one on different runs. So
# both look the same on screen, and the fallback below uses this dict too — the
# same outcome reads the same whichever method produced it.
VERDICT_WORDING = {
    "grounded": "### ✅ Supported by your source",
    "hallucinated": "### ⚠️ Not supported by your source",
}

# Measured on the 300-case test split by analyze.py: accuracy, precision,
# recall, F1. Written as numbers rather than computed at start-up, because the
# cache they come from is not in the repository — the Space has no way to
# recompute them, and a page that re-scored 300 cases on every visit would be
# the wrong design even if it could.
ACCURACY = {"embeddings": 0.357, "entailment": 0.640, "judge": 0.817}
BENCHMARK = {
    "Text similarity<br>embeddings<br>all-MiniLM-L6-v2":
        (0.357, 0.235, 0.127, 0.165),
    "Entailment<br>HHEM-2.1-Open": (0.640, 0.750, 0.420, 0.538),
    "AI judge<br>gpt-oss-120b": (0.817, 0.832, 0.793, 0.812),
}

# "grounded" and "hallucinated" are our internal words. On screen a visitor
# reads plain ones, and the same two words are used for every method so the
# rows of the breakdown can be compared at a glance.
OUTCOME = {"grounded": "no error found", "hallucinated": "error found"}


# The limit is set by memory, not by any model's context window. HHEM reads a
# long document in full — its own loading code calls the tokenizer without
# truncation — and attention costs grow faster than the text: measured on this
# machine, one check takes 1.2s and 2.0 GB at 10,000 characters, 2.2s and 4.7 GB
# at 15,000, and 8.0s and 6.2 GB at 25,000. The Space has 16 GB with 2-3 GB
# already held by torch and the two models, so two visitors checking 25,000
# characters at once would take it down, and a Space that dies on Demo Day is
# the worst outcome available. 10,000 leaves room for three at a time.
#
# Groq is not the binding limit at this length, though it looked like it at
# first: 10,000 characters is roughly 2,100 tokens against a per-minute
# allowance of 8,000, so an over-long request can no longer reach it.
MAX_CHARS = 10_000

# Stated in words as well, because nobody counts characters. 1,660 by the 6.03
# characters per word measured across all 380 sources in this project's own
# data; rounded down, so the hint never promises more than it means.
MAX_WORDS = "1,600"

# Applied to whichever field a refused check pointed at, and removed from the
# other. Kept as names rather than written inline: the marker is set in three
# places and read by one CSS rule, and a typo in a class name fails silently.
PLAIN: list[str] = []
OVER = ["over-limit"]


def field_marks(flagged: str = "") -> tuple[dict, dict]:
    """Class updates for the two input fields; at most one carries the marker."""
    return (gr.update(elem_classes=OVER if flagged == "source" else PLAIN),
            gr.update(elem_classes=OVER if flagged == "answer" else PLAIN))


def length_note(source: str, response: str) -> str:
    """The live count under the fields, and the only warning about length.

    One count for both fields, because the limit applies to a check as a whole:
    two separate counters would imply each field had its own budget. Rendered
    even when both are empty, so the line holds its height and the button does
    not shift down the moment typing starts.
    """
    used = len(source) + len(response)

    if not used:
        return (f"*Up to {MAX_CHARS:,} characters across both fields — about "
                f"{MAX_WORDS} words*")

    if used <= MAX_CHARS:
        return f"*{used:,} of {MAX_CHARS:,} characters*"

    # Name the field that is actually long. In practice it is the document — an
    # AI's answer is a paragraph or two — but a pasted ten-page answer would
    # make a hard-coded guess wrong, and wrong about which text to cut.
    field = "your document" if len(source) >= len(response) else "the AI's answer"
    return (f"*⚠️ {used:,} of {MAX_CHARS:,} characters. Remove about "
            f"{used - MAX_CHARS:,} from {field} — this count updates as you "
            "cut.*")


def breakdown_table(embedding=None, entailment_result=None, judge=None,
                    judge_unavailable=False) -> str:
    """The three-method table, with a dash wherever a check has not run.

    The first two columns hold before anything is checked — what each method is
    and how often it is right in general — so the table doubles as the panel's
    empty state. A visitor reads what the product does before pressing anything,
    and no space sits blank to no purpose.
    """
    dash = '<span class="empty">—</span>'

    def score_cell(result, threshold: float, gloss: str) -> str:
        """The score, where it falls, and what the number means.

        A number alone says nothing here: each method starts calling an answer
        unsupported at a different place, and neither place is where a reader
        would guess. The bar is filled to the score with a tick at that
        method's own cut-off, so the cell answers "high or low" by itself and
        never invites a comparison with the row above.
        """
        tick = f'<span class="tick" style="left:{threshold * 100:.0f}%"></span>'

        if result is None:
            # Before anything runs the track still shows where the line is.
            return f'{dash}<span class="bar">{tick}</span>{gloss}'

        # Clamped because cosine similarity can be negative — four of the 300
        # test answers scored below zero — and a negative width draws nothing.
        width = min(max(result.score, 0.0), 1.0) * 100
        fill = f'<span class="fill" style="width:{width:.0f}%"></span>'
        return f'{result.score:.2f}<span class="bar">{fill}{tick}</span>{gloss}'

    def cells(result):
        if result is None:
            return dash, dash, dash
        return (OUTCOME[result.verdict], f"{result.score:.2f}",
                f"{result.latency_ms:.0f} ms")

    sim_v, sim_s, sim_t = cells(embedding)
    ent_v, ent_s, ent_t = cells(entailment_result)

    # Built here rather than inside the table below: the gloss carries an
    # apostrophe, and quoting it inside an f-string inside a table row is the
    # kind of line nobody can edit later without breaking it.
    sim_cell = score_cell(embedding, embeddings.threshold,
                          "(how much the answer's wording matches your document)")
    ent_cell = score_cell(entailment_result, entailment.threshold,
                          "(how sure the answer follows from your document)")
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
        f"| Text similarity<br>embeddings<br>all-MiniLM-L6-v2 | Compares "
        f"overall wording, not facts, and reads only the start of a long "
        f"document. Right on {ACCURACY['embeddings']:.0%} (accuracy). "
        f"| {sim_v} | {sim_cell} | {sim_t} |",
        f"| Entailment<br>HHEM-2.1-Open | Asks whether the answer follows from "
        f"the document. Right on {ACCURACY['entailment']:.0%} (accuracy). "
        f"| {ent_v} | {ent_cell} | {ent_t} |",
        f"| AI judge<br>gpt-oss-120b | Reads both and says what it thinks is "
        f"unsupported. Right on {ACCURACY['judge']:.0%} (accuracy). "
        f"| {judge_v} | {dash}<br>(rules, does not score) | {judge_t} |",
        "",
        "*The accuracy percentages come from 300 answers in HaluEval, a public "
        "set in which people marked which AI answers were faithful to their "
        "source. Only those are comparable between methods; the scores are not.*",
    ])

# ---- The second tab: how the three methods were measured -------------------

BENCHMARK_INTRO = """### Where these numbers come from

All three methods were measured on **HaluEval**, a public set in which people
marked whether an AI's answer was faithful to its source. 150 records gave one
faithful answer and one hallucinated each — the same **300 answers** for every
method.

**Accuracy** counts every kind of mistake together. **Recall** is the share of
real errors a method found, and it matters most here: an error that reaches you
unflagged is what this product exists to prevent. **Precision** is how often an
alarm was real, and **F1** combines the two."""

METHODS_CHART_NOTE = """A weaker method is not simply wrong more often. It
fails in one direction: text similarity is right about 36% of answers and finds
only 13% of the errors. Entailment rarely raises a false alarm — three out of
four answers it flags really are wrong — but it stays silent on more than half
of them."""

CASCADE_TEXT = """### Could a cheap method do the work instead?

Only where it is already sure. Entailment (HHEM-2.1-Open) scores every answer
from 0 to 1, and a very low score — under 0.2 — means it has already found the
problem. On those answers the AI judge (gpt-oss-120b) is right exactly as
often, so asking it changes nothing.

So the cheap method answers the ones it is sure about and the AI judge reads
the rest: 82% of answers, the same accuracy as sending it everything, and 18%
less cost.

Spending less is free only up to a point: removing the last fifth of the cost
changes nothing, and paying half as much costs ten points of accuracy."""


def benchmark_table() -> str:
    """One row per method, four measures each, in the order of the intro above."""
    rows = ["| Method | Accuracy | Precision | Recall | F1 |",
            "|---|---|---|---|---|"]

    for name, (accuracy, precision, recall, f1) in BENCHMARK.items():
        rows.append(f"| {name} | {accuracy:.3f} | {precision:.3f} | "
                    f"{recall:.3f} | {f1:.3f} |")

    return "\n".join(rows)



def check(source: str, response: str) -> tuple:
    """Run all three detectors and phrase the outcome for a reader.

    Returns four things: the verdict a visitor reads, the method-by-method table
    behind it, and a class update for each input field — the marker goes on the
    one a refused check pointed at, and comes off both otherwise.
    """

    # Run on two empty strings the detectors agree enthusiastically — the cosine
    # similarity of nothing with nothing is 1.00 — and the screen then asserted
    # that an empty answer was supported by an empty document. Checking first
    # also saves a judge call, which costs quota.
    if not source.strip() or not response.strip():
        return ("*Nothing to check yet — paste a document and an answer first.*",
                breakdown_table(), *field_marks())

    # Over the limit nothing runs at all — not the judge, whose free allowance
    # would be spent on a check that cannot be honest, and not the two local
    # detectors, whose memory is the reason the limit exists. Two alternatives
    # were rejected on the way. Shortening the document and checking the rest
    # would report a claim supported on a later page as unsupported, which is
    # the product doing the very thing it exists to catch. Splitting it into
    # parts and sending one a minute costs six calls and six minutes for a long
    # document, about 17% of the day's allowance on a key shared by every
    # visitor, and it would invalidate the accuracy figures on screen, which
    # were measured on whole documents.
    used = len(source) + len(response)
    if used > MAX_CHARS:
        longer = "source" if len(source) >= len(response) else "answer"
        return (
            "### ⚠️ Too long to check\n\n"
            f"*This checks up to {MAX_CHARS:,} characters — about {MAX_WORDS} "
            f"words — and this is {used:,}. Remove about {used - MAX_CHARS:,} "
            "characters, or check one section of your document at a time.*",
            breakdown_table(), *field_marks(longer),
        )

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
    # outside service on a free tier that runs out, and it needs a key. Each
    # reason gets its own sentence — "come back later" is sound advice when the
    # allowance is spent and a lie when the key is missing.
    judge_result = None
    judge_note = ""
    tail = ("The verdict below comes from the entailment check instead: it "
            "answers yes or no, with no explanation of either.*")
    if judge is None:
        judge_note = (
            "*The AI judge did not run: it is not configured in this "
            "deployment, and retrying will not change that. " + tail
        )
    else:
        try:
            judge_result = judge.check(case)
        except RateLimitError:
            judge_note = (
                "*The AI judge did not run: its free daily allowance is used "
                "up, and it resets at midnight UTC. " + tail
            )
        except Exception as error:
            # Logged rather than displayed: the visitor needs the sentence
            # below, the cause belongs in the Space's log.
            print(f"judge failed: {error}")
            judge_note = (
                "*The AI judge could not be reached just now. " + tail
            )

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
                    "\n*The two methods disagree: the entailment check reads this "
                    "answer as supported. What the judge objected to is above — "
                    "check it against your document yourself.*"
                )
            else:
                lines.append(
                    "\n*The two methods disagree: the AI judge found nothing "
                    "unsupported, while the entailment check was not convinced. "
                    "Nothing specific was named — worth reading the answer against "
                    "your document yourself.*"
                )

    else:
        # Fallback: entailment decides. The note goes first on purpose — a
        # verdict read before its provenance reads as the product's own word,
        # and this is not the verdict the product normally gives.
        lines.append(judge_note)
        lines.append("\n" + VERDICT_WORDING[entailment_result.verdict])

    # One row per method, always on screen rather than hidden behind a disclosure:
    # comparing the three is the point of the project, and a collapsed panel also
    # left a hole in the layout. The columns answer three different questions —
    # what the method is, what it said about these two texts, and how often it is
    # right in general. The last must not read as a property of this check, which
    # is why it has its own column and the footnote below.
    return (
        "\n".join(lines),
        breakdown_table(embedding_result, entailment_result, judge_result,
                        judge_unavailable=judge_result is None),
        *field_marks(),
    )


def clear_result(source: str, response: str) -> tuple:
    """Reset everything a result was true of, and recount the characters.

    Never sets the over-limit marker: someone still typing has not asked for
    anything yet, so the warning at this stage is the count alone. The marker is
    set only by a press, and cleared here so the screen cannot hold a red field
    beside a count that is back inside the limit.
    """
    return ("", breakdown_table(), length_note(source, response), *field_marks())


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
#length-line .md.prose,
#breakdown-panel .md.prose,
#breakdown-panel table {
  font-size: var(--block-info-text-size);
  line-height: var(--line-sm);
}
#when-line .md.prose *,
#result-caption .md.prose *,
#length-line .md.prose *,
#breakdown-panel .md.prose * { color: var(--block-info-text-color); }

/* The field a refused check pointed at. Set on a press, never while typing, so
   it marks a refusal rather than nagging at someone mid-paste. Keyed on the id
   because Gradio draws the field's own border through a class of its own, and
   its stylesheet loads last. */
#source-field.over-limit textarea,
#answer-field.over-limit textarea {
  border-color: #b91c1c;
  box-shadow: 0 0 0 1px #b91c1c;
}

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
.gradio-container { width: 100%; max-width: 1060px; margin-inline: auto; }
gradio-app { display: block; width: 100%; }

/* The container is capped and sits flush left, so on a wider window its right
   edge left an uncovered strip showing white through. The tint has to sit on
   the document itself. `gradio-app` needs !important for the same reason the
   width rule does: Gradio writes its background as an inline style, and no
   stylesheet rule outranks one. */
body { background: #f1f2f4; }
gradio-app { background: #f1f2f4 !important; }
/* !important is not decoration here: Gradio writes this cap as an inline style
   on <main>, and no stylesheet rule outranks an inline style without it. */
/* Width is taken from the window, never from the content. Gradio's fillable
   mode sizes this container to whatever the visible tab happens to contain, so
   switching to the comparison tab — whose text is capped at 460 and charts at
   520 — shrank the whole page from 1440 to 1060, moving the header and the tab
   bar with it. Fixing the width holds everything still. */
body .gradio-container main.app.fillable {
  width: 100% !important;
  max-width: 1060px !important;
}

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
#result-card .styler { padding: 0 12px 12px 12px; }

/* "Audit verdict" sits on the same line as "Your document", and the caption on
   the same line as that field's hint. Measured: the card's first block starts
   10px higher than the form's first label. */
#result-label { margin-top: 10px; }
#result-label .md.prose * {
  color: var(--block-title-text-color);
  font-size: var(--block-title-text-size);
  font-weight: var(--block-title-text-weight);
}
/* Matched to the left column by measurement, not by eye: Gradio sets its
   field hint at 17.875px of line and 2px below the label, and the caption
   opposite has to do the same or the two columns' second lines sit apart. */
#result-caption { margin-top: 1px; margin-bottom: 5px; }
#result-caption .md.prose p { line-height: 17.875px; }

/* The panel keeps its size whatever it holds: a result that resized the page
   moved the button and the examples under the reader's cursor. Taller content
   scrolls inside instead. Height is the left column measured from the label to
   the bottom of the button. */
/* Gradio's own footer — "Use via API · Built with Gradio · Settings" — sits
   below the examples and, with the space reserved around it, adds 51px to a
   page that has to fit one screen. Nothing on it belongs to this product, and
   removing it is what keeps the whole thing visible without scrolling on a
   717px-tall window.

   !important on both: Gradio styles its footer through `footer` plus two of
   its own generated classes, which outranks a plain element selector, and the
   page padding is set the same way. This is the fourth time in this file that
   a rule which looked correct did nothing until it outranked Gradio's own. */
footer { display: none !important; }
main.app.fillable { padding-bottom: 0 !important; }

/* The right column's heading and caption are Markdown, the left column's are
   Gradio's own label and hint. Same face, same size, same top — but Markdown
   carries a taller line, so the two columns' text sat at different heights
   inside matching rows, and the difference showed the moment either wrapped.
   1.4 is what the left column uses. */
#result-label .md.prose p { line-height: 1.4; }

/* The claim the judge flagged needs the same air under it that the table has
   under it — 17px, measured, not picked. As padding on the panel rather than
   a margin on the paragraph: Gradio's own typography sets `.prose :last-child`
   to zero, and a margin there loses or collapses away whatever it is set to.
   :has() limits it to a panel that actually holds a verdict. */
#verdict-panel:has(h3) { padding-bottom: 17px; }

/* Above it, the opposite problem: Gradio puts 16px over every paragraph that
   is not the first, which is a lot between a verdict and the claim under it. */
#verdict-panel .md.prose p { margin-top: 6px; }

/* Empty, the panel still costs a row gap, and the table opposite then starts
   6px below the field it should line up with. Removed from the flow entirely
   until there is something in it. */
#verdict-panel:not(:has(.md.prose > *)) { display: none; }

/* The example buttons come in at 13px, the size of a field label, while the
   text around them sits at 11px — so they read as the loudest thing in the
   lower half of the screen for something that is a convenience, not the task.
   Dropped to the supporting size already in the scale rather than a new one. */
.gallery-item, .gallery-item button { font-size: var(--block-info-text-size); }

/* The tab bar costs 48px of a page that has to fit one screen, and the header
   above it was spending 64 of its 155 on air. The gaps between the header
   blocks are Gradio's own row-gap, which cannot be reached without matching
   its generated class names — so the space is taken back as negative margins
   on the blocks themselves, which is both reachable and exact. */
#page-title h1 { font-size: 1.3rem; }
#page-title, #intro-line, #when-line { margin-bottom: -8px; }
main.app.fillable { padding-top: 10px !important; }

/* The charts are drawn at 1650px wide for a projector. Left at that size they
   fill the screen one at a time; at 520 both sit beside the text about them. */
#methods-chart img, #tradeoff-chart img { max-width: 430px; }

/* Gradio centres an image inside its block. Beside a column of text that
   starts at the edge, a centred chart reads as indented by accident — and the
   lower chart sits under the table it belongs to, where the two left edges
   have to agree. The centring lives on the button Gradio wraps the image in,
   not on the block: a rule on the block changed nothing. */
#methods-chart button, #tradeoff-chart button { justify-content: flex-start; }

/* The second tab arrived using the browser's default sizes, two steps larger
   than anything on the first tab. It takes the same scale: 11px for prose and
   for the table, and a heading one step up rather than three. This column is
   explanation, the same role the 11px hints play on the first tab, so it reads
   at the same size rather than introducing a step of its own. At 13px it also
   ran to within 17px of the bottom of a 770px window; at 11px it stops 126px
   short. */
#compare-text .md.prose,
#compare-cascade .md.prose,
#compare-chart-top .md.prose {
  /* Gradio renders this as a span, and a max-width on an inline element does
     nothing at all — the rule below looked correct and changed no pixel. */
  display: block;
  font-size: 11px;
  line-height: 1.5;
  /* The measure is held at 67 characters, inside the 45-75 a reader holds
     comfortably: the column is wider than that, so the cap does the work, and
     it moves with the font size — 460 at 13px, 389 at 11px. */
  max-width: 389px;
}
#compare-text .md.prose h3,
#compare-cascade .md.prose h3 { font-size: 15px; font-weight: 700; margin: 0 0 6px 0; }
#compare-text .md.prose p,
#compare-cascade .md.prose p,
#compare-chart-top .md.prose p { margin: 6px 0 10px 0; }

/* Dark is what a reader must read — the verdict, their own pasted text. Text
   that explains is grey, on both tabs, so the rule holds wherever it appears. */
#compare-text .md.prose p, #compare-cascade .md.prose p,
#compare-chart-top .md.prose p, #intro-line .md.prose p {
  color: var(--block-info-text-color);
}
/* Every 11px element in this app is grey; this table arrived near-black and
   was the one thing on the second tab that broke the scale. */
#benchmark-table .md.prose, #benchmark-table table,
#benchmark-table td, #benchmark-table th {
  font-size: 11px;
  color: var(--block-info-text-color);
}

/* A score on its own says nothing: each method starts calling an answer
   unsupported at a different place, and neither place is where a reader would
   guess. The bar is filled to the score with a tick at that row's own cut-off,
   so the cell shows how far from the line the number sits — which a colour
   could not — and the ticks standing in different places keep the three rows
   from reading as one shared scale. Rejected on the way: colouring the number,
   which would have repeated the Verdict column beside it. */
#breakdown-panel .bar {
  position: relative;
  display: block;
  width: 100%;
  max-width: 120px;
  height: 4px;
  margin: 3px 0;
  border-radius: 2px;
  background: #e4e4e7;
}
#breakdown-panel .bar .fill {
  position: absolute;
  left: 0; top: 0; bottom: 0;
  border-radius: 2px;
  background: #9ca3af;
}
#breakdown-panel .bar .tick {
  position: absolute;
  top: -2px; bottom: -2px;
  width: 1px;
  background: #3f3f46;
}

/* Digits share one width down the numeric columns. Only the figures change —
   the face stays the body face, because those columns now carry prose as well
   as numbers, and two typefaces in one small table read as an accident. */
#breakdown-panel td:nth-child(n+4) {
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
#breakdown-panel th:nth-child(3), #breakdown-panel td:nth-child(3),
#breakdown-panel th:nth-child(5), #breakdown-panel td:nth-child(5) {
  white-space: nowrap;
}

/* The method column carries two lines by design — the name a visitor reads and
   the model behind it. Without a floor the table hands its width to the other
   columns and breaks the model name across four lines. Measured: 140px is what
   keeps each on one line of its own. */
#breakdown-panel th:nth-child(1), #breakdown-panel td:nth-child(1) {
  min-width: 110px;
}

/* Gradio pads table cells for a page with room to spare. This one sits in a
   fixed panel beside a column of inputs, and the padding was costing it both
   the height it needed and the width that stopped it wrapping. */
#breakdown-panel td, #breakdown-panel th { padding: 4px 6px; }

/* The same reasoning one tab over: this table sets the height of the top row
   of the grid, and Gradio's 9px of vertical cell padding was four rows of it.
   The horizontal 13px stays — columns of numbers need the air. */
#benchmark-table td, #benchmark-table th { padding: 5px 13px; }

/* The outline was being drawn twice — once by the table and once by the cells
   along its edge — and with collapsed borders the two land a fraction of a
   pixel apart, which shows as a notch at every row on the left edge. The cells
   draw the whole grid; the table draws nothing. */
/* The grid was drawn three times over: a border on the table, another on every
   row, a third on every cell — and 2px of spacing holding them apart, so the
   horizontal lines stopped short of the vertical one and every corner showed a
   notch. Collapsed, with the cells the only thing drawing, the grid closes. */
#breakdown-panel table, #benchmark-table table {
  /* One outline, drawn once, by the table itself. The cells below draw only
     the lines between them, so nothing meets anything at a corner. */
  border: 1px solid var(--border-color-primary);
  border-collapse: collapse;
  border-spacing: 0;
}
#breakdown-panel tr, #benchmark-table tr { border: none; }

/* Gradio wraps its tables in a container that already draws a rounded outline.
   The cells were drawing a second, square one just inside it, and a square
   corner cannot meet a rounded one — which is the notch visible at every row
   on the left edge. The cells now draw only the lines between themselves and
   stop at the edges, leaving the outline to the wrapper that owns it. */
#breakdown-panel td, #breakdown-panel th,
#benchmark-table td, #benchmark-table th {
  border: none;
  border-bottom: 1px solid var(--border-color-primary);
  border-right: 1px solid var(--border-color-primary);
}
#breakdown-panel tr:last-child td, #benchmark-table tr:last-child td {
  border-bottom: none;
}
#breakdown-panel td:last-child, #breakdown-panel th:last-child,
#benchmark-table td:last-child, #benchmark-table th:last-child {
  border-right: none;
}


#result-body {
  /* Fixed rather than fluid so the card ends level with the button opposite
     it. Measured in a browser, not guessed: 465 is what puts both columns at
     the same bottom now that the page is capped at 1060 and split
     34/66, which is what the breakdown table needs to stop wrapping. */
  height: 443px;
  overflow-y: auto;
  display: block;
}

/* Three sizes on the page, and only three. The verdict is the answer, so it is
   the one piece of display type. 13px carries prose the visitor has to read —
   the description at the top, the claim the judge flagged. Everything that
   explains rather than answers sits at 11px in grey: field hints, the pasted
   document, this panel's caption, the table and every aside.
   The verdict is emitted as a heading rather than bold text. An earlier rule
   keyed on "bold in the first paragraph" and silently stopped matching the day
   a note was placed above the verdict, which shrank it to body size. */
#verdict-panel .md.prose h3 {
  font-size: 1.3rem;
  font-weight: 700;
  margin: 0 0 var(--spacing-md) 0;
}

/* Italic marks an aside: the missing-judge note, the disagreement note, the
   footnote under the table. They take the supporting size, not body size. */
#verdict-panel .md.prose em {
  color: var(--block-info-text-color);
  font-size: var(--block-info-text-size);
  line-height: var(--line-sm);
}
"""

with gr.Blocks(
    title="AI Output Auditor",
    # primary_hue sets the colour of the primary button. The default orange
    # shouts on a page whose whole subject is calibrated confidence; slate
    # still reads as the one action to take, without the alarm.
    # Inter is drawn for screens at small sizes, and this page is mostly small
    # text: two field hints, a caption and a footnote at 11px. The breakdown
    # table takes the same face throughout, with tabular figures where numbers
    # need to line up.
    theme=gr_themes.Default(
        text_size=gr_themes.sizes.text_sm,
        primary_hue=gr_themes.colors.slate,
        font=gr_themes.GoogleFont("Inter"),
        font_mono=gr_themes.GoogleFont("IBM Plex Mono"),
    ),
    css=CSS,
) as demo:
    gr.Markdown("# AI Output Auditor", elem_id="page-title")
    gr.Markdown(
        elem_id="intro-line",
        value="Gave an AI a document and asked it something? Paste both below — "
        "this checks the answer against your document and says what it does not "
        "back up."
    )
    gr.Markdown(
        "Most worth doing when you asked for figures, dates or names, when the answer "
        "may go beyond what your document covers, or when the document is long.",
        elem_id="when-line",
    )

    # Two surfaces, not one page: a visitor with a document wants a verdict,
    # and the benchmark behind it is a different question asked by a different
    # reader. Tabs keep the second from crowding the first.
    with gr.Tabs():
        with gr.Tab("Check an answer"):

            # A Row places its children side by side; a Column stacks them. equal_height
            # is off on purpose: it forces both columns to the same height and hands the
            # spare space to every child, which stretched the button whenever the result
            # grew and left a gap above the breakdown when it did not.
            with gr.Row(equal_height=False):
                with gr.Column(elem_id="input-card", scale=42):
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
                    # Rendered from the start rather than appearing on first use: the
                    # line reserves its own height, so the button below cannot shift
                    # down under the cursor the moment someone starts typing.
                    length_line = gr.Markdown(length_note("", ""), elem_id="length-line")
                    check_button = gr.Button("Check the answer", variant="primary")

                with gr.Column(scale=58):
                    with gr.Group(elem_id="result-card"):
                        gr.Markdown("Audit verdict", elem_id="result-label")
                        # Static: true of every check, so it does not wait for one.
                        gr.Markdown(
                            "All three methods check the answer.<br>Only the AI "
                            "judge explains what it found, so the verdict comes "
                            "from it.",
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
                outputs=[verdict_box, breakdown_box, source_box, response_box],
            )

            # A result is only true of the inputs it was computed from. The moment either
            # one changes — including when clicking an example fills them — the panel is
            # cleared, so the screen can never show a new document beside an old verdict.
            # This is why the button stays: running on every keystroke would spend quota
            # on half-pasted text and give the visitor no say in sending their document
            # to an outside service.
            # The two fields are outputs as well as inputs here, but only ever receive a
            # class update — no value is sent back, so what is being typed is untouched.
            for box in (source_box, response_box):
                box.change(fn=clear_result,
                           inputs=[source_box, response_box],
                           outputs=[verdict_box, breakdown_box, length_line,
                                    source_box, response_box])

            # Examples must be created explicitly now, and told which components they
            # fill. One is a benchmark answer written to contain an error, the other a
            # real model answer that holds up; both are documented, with their sources
            # and licences, in examples.py.

            gr.Examples(
                examples=EXAMPLES,
                inputs=[source_box, response_box],
                example_labels=EXAMPLE_LABELS,
            )

        with gr.Tab("How the methods compare"):
            # A 2x2 grid rather than two tall columns. Stacked in one page
            # this tab ran to 1,531px and set its prose across the full 1,440 —
            # about 180 characters a line, where a reader loses the start of
            # the next one. In two columns the charts ended up in a stack of
            # their own, far from the sentences that read them. Here each chart
            # sits beside its own text and the sides alternate, so the eye
            # crosses the page once per row instead of running down one column
            # and back up the other.
            #
            # The charts are files in the repository rather than drawings made
            # here: the same two images go into the deck and the README, and
            # three renderings of one measurement would drift.
            with gr.Row(equal_height=False):
                with gr.Column(scale=45, elem_id="compare-text"):
                    gr.Markdown(benchmark_table(), elem_id="benchmark-table")
                    gr.Markdown(BENCHMARK_INTRO)

                with gr.Column(scale=55, elem_id="compare-chart-top"):
                    gr.Image("charts/methods.png", show_label=False,
                             container=False, show_download_button=False,
                             interactive=False, elem_id="methods-chart")
                    gr.Markdown(METHODS_CHART_NOTE)

            with gr.Row(equal_height=False):
                with gr.Column(scale=45, elem_id="compare-chart-bottom"):
                    gr.Image("charts/tradeoff.png", show_label=False,
                             container=False, show_download_button=False,
                             interactive=False, elem_id="tradeoff-chart")

                with gr.Column(scale=55, elem_id="compare-cascade"):
                    gr.Markdown(CASCADE_TEXT)
        


# SSR is on by default and shuts the app down immediately on Spaces — a known
# Gradio issue, unrelated to ZeroGPU.
demo.launch(ssr_mode=False)
