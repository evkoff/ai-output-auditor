"""Worked examples shown under the interface: one with an error, one without.

The first shows the check finding an error, the second shows it raising no
false alarm. Each button says where its answer came from, because the field it
fills is labelled "The AI's answer" — a visitor reads anything placed there as
something an AI wrote, and a note in this file would never reach them.

The texts are written as adjacent string literals inside parentheses. Python
joins them with nothing in between, so these lines can be rewrapped freely
without changing a character. Inside triple quotes that is not true: every line
break becomes part of the text, and would change what the detectors read.

1. FIRE — the answer contains an error. From HaluEval (MIT licence,
   https://github.com/RUCAIBox/HaluEval), summarization subset, test split.
   The article is CNN reporting, carried into HaluEval through the CNN/Daily Mail
   dataset. It is quoted here as a short excerpt to demonstrate the tool;
   copyright in the article remains with its publisher.
   ⚠️ The answer is NOT a model's natural mistake. HaluEval produced its
   hallucinated summaries deliberately: ChatGPT was instructed to write one, and
   the most plausible were kept. The button label says so.

2. RASPBERRY PI — the answer has no errors. Wikipedia lead section, CC BY-SA 4.0,
   https://en.wikipedia.org/w/index.php?title=Raspberry_Pi&oldid=1373778583
   The answer is Gemini 3.6 Flash's, given the instruction "Answer the question
   using only the article below."
"""

# The article says the authorities "didn't know what had caused the fire"; the
# answer names a cause. The judge and entailment both flag it; text similarity
# calls it grounded, which is its 36% accuracy in action.
FIRE_ARTICLE = (
    "A mammoth fire broke out Friday morning in a Kentucky industrial park, "
    "sending plumes of thick smoke over the area as authorities worked to "
    "contain the damage. The blaze began shortly before 7 a.m. at the "
    "General Electric Appliance Park in Louisville, according to Mike Weimer "
    "from the city's emergency management agency. He said that there were no "
    "reports of anyone injured or trapped. Video showed both smoke and "
    "bright orange flames. Firefighters took up positions around the "
    "affected buildings, spraying water from the periphery. Weimer told CNN "
    "that authorities didn't know what had caused the fire, which had gone "
    "to at least four alarms. According to a GE website, its facility in the "
    "Louisville Appliance Park is \"revitalizing manufacturing in the United "
    "States.\" The park is large, such that 34 football fields could fit in "
    "one of its warehouses in the facility."
)

FIRE_ANSWER = (
    "The fire in Louisville's General Electric Appliance Park was caused by "
    "a malfunctioning robot, which caused significant damage but no "
    "injuries. The park's large warehouses were severely impacted, but "
    "authorities are confident that production will continue as usual."
)

# Every claim is supported, so the screen shows what "nothing wrong" looks like.
RASPBERRY_PI_ARTICLE = (
    "Raspberry Pi ( PY) is a series of small single-board computers (SBCs) "
    "originally developed in the United Kingdom by the Raspberry Pi "
    "Foundation in collaboration with Broadcom. To commercialize the product "
    "and support its growing demand, the Foundation established a commercial "
    "entity, now known as Raspberry Pi Holdings.\n"
    "The Raspberry Pi was originally created to help teach computer science "
    "in schools, but gained popularity for many other uses for its low cost, "
    "high performance, compact size, and flexibility. It is now used in "
    "areas such as industrial automation, robotics, home automation, IoT "
    "devices, and hobbyist projects.\n"
    "The company's products range from simple microcontrollers to computers "
    "that the company markets as being powerful enough to be used as a "
    "general purpose PC. Computers are built around a custom designed system "
    "on a chip and offer features such as HDMI video/audio output, USB "
    "ports, wireless networking, GPIO pins, and up to 16 GB of RAM. Storage "
    "is typically provided via microSD cards.\n"
    "In 2015, the Raspberry Pi surpassed the ZX Spectrum as the best-selling "
    "British computer of all time. As of March 2026, over 75 million units "
    "had been sold."
)

RASPBERRY_PI_ANSWER = (
    "Based on the provided article:\n"
    "\n"
    "Original Purpose: The Raspberry Pi was originally created to help teach "
    "computer science in schools.\n"
    "\n"
    "Units Sold: Over 75 million units had been sold as of March 2026."
)

EXAMPLES = [
    [FIRE_ARTICLE, FIRE_ANSWER],
    [RASPBERRY_PI_ARTICLE, RASPBERRY_PI_ANSWER],
]

# Shown on the buttons instead of the full text. No case identifiers: they mean
# nothing to a visitor, and the provenance above is where they belong.
EXAMPLE_LABELS = [
    "Fire at a GE plant — test answer, made-up cause",
    "Raspberry Pi — a real AI answer with no errors",
]
