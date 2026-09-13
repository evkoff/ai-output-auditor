"""Worked examples shown under the interface.

Both are real: a Wikipedia article, and the answer an actual model gave when
asked to use only that article. Toy sentences were rejected deliberately — they
show the tool catching mistakes no current model makes, which misrepresents both
the product and the problem.

Model: Gemini 3.6 Flash (gemini.google.com app).
Instruction given to it: "Answer the question using only the article below."

Article text: Wikipedia, quoted verbatim from the lead section, CC BY-SA 4.0.
  Apollo 13    — https://en.wikipedia.org/w/index.php?title=Apollo_13&oldid=1369194288
  Raspberry Pi — https://en.wikipedia.org/w/index.php?title=Raspberry_Pi&oldid=1373778583

Both cases are labelled in data/real/ (not in git): the Apollo answer is
hallucinated, the Raspberry Pi answer is grounded.
"""

# The article says the crew "looped around the Moon in a circumlunar trajectory".
# The model's summary says they "orbited the Moon" — a different manoeuvre, and
# one Apollo 13 never performed. Subtle, plausible, and visible only against the
# source: the case the product exists for.
APOLLO_ARTICLE = """\
Apollo 13 (April 11–17, 1970) was the seventh crewed mission in the Apollo space program and would have been the third Moon landing. The craft was launched from Kennedy Space Center on April 11, 1970, but the landing was aborted after an oxygen tank in the service module (SM) exploded two days into the mission, disabling its electrical and life-support system. The crew, supported by backup systems on the Apollo Lunar Module, instead looped around the Moon in a circumlunar trajectory and returned safely to Earth on April 17. The mission was commanded by Jim Lovell, with Jack Swigert as command module (CM) pilot and Fred Haise as Lunar Module (LM) pilot. Swigert was a late replacement for Ken Mattingly, who was grounded after exposure to rubella.
A routine stir of an oxygen tank ignited damaged wire insulation inside it, causing an explosion that vented the contents of both of the SM's oxygen tanks to space. Without oxygen, needed for breathing and for generating electrical power, the SM's propulsion and life support systems could not operate. The CM's systems had to be shut down to conserve its remaining resources for reentry, forcing the crew to transfer to the LM as a lifeboat. With the lunar landing canceled, mission controllers worked to bring the crew home alive.
Although the LM was designed to support two men on the lunar surface for two days, Mission Control in Houston improvised new procedures so it could support three men for four days. The crew experienced great hardship, caused by limited power, a chilly and wet cabin and a shortage of drinking water. There was a critical need to adapt the CM's cartridges for the carbon dioxide scrubber system to work in the LM; the crew and mission controllers were successful in improvising a solution. The astronauts' peril briefly renewed public interest in the Apollo program; tens of millions watched the splashdown in the South Pacific Ocean on television.
An investigative review board found fault with preflight testing of the oxygen tank and Teflon being placed inside it. The board recommended changes, including minimizing the use of potentially combustible items inside the tank; this was done for Apollo 14. The story of Apollo 13 has been dramatized several times, most notably in the 1995 film Apollo 13 (based on Lost Moon, the 1994 memoir co-authored by Lovell) and an episode of the 1998 miniseries From the Earth to the Moon."""

APOLLO_ANSWER = """\
Apollo 13 launched on April 11, 1970, but its planned Moon landing was canceled after an oxygen tank explosion disabled the Service Module's electrical and life-support systems. To survive the return journey, the three-man crew used the Lunar Module as an improvised lifeboat while Mission Control developed emergency procedures to stretch its limited resources. Supported by these improvisations, the astronauts successfully orbited the Moon and returned safely to Earth, splashing down on April 17."""

# A clean answer, so the interface also shows what "nothing wrong" looks like.
RASPBERRY_PI_ARTICLE = """\
Raspberry Pi ( PY) is a series of small single-board computers (SBCs) originally developed in the United Kingdom by the Raspberry Pi Foundation in collaboration with Broadcom. To commercialize the product and support its growing demand, the Foundation established a commercial entity, now known as Raspberry Pi Holdings.
The Raspberry Pi was originally created to help teach computer science in schools, but gained popularity for many other uses for its low cost, high performance, compact size, and flexibility. It is now used in areas such as industrial automation, robotics, home automation, IoT devices, and hobbyist projects.
The company's products range from simple microcontrollers to computers that the company markets as being powerful enough to be used as a general purpose PC. Computers are built around a custom designed system on a chip and offer features such as HDMI video/audio output, USB ports, wireless networking, GPIO pins, and up to 16 GB of RAM. Storage is typically provided via microSD cards.
In 2015, the Raspberry Pi surpassed the ZX Spectrum as the best-selling British computer of all time. As of March 2026, over 75 million units had been sold."""

RASPBERRY_PI_ANSWER = """\
Based on the provided article:

Original Purpose: The Raspberry Pi was originally created to help teach computer science in schools.

Units Sold: Over 75 million units had been sold as of March 2026."""

EXAMPLES = [
    [APOLLO_ARTICLE, APOLLO_ANSWER],
    [RASPBERRY_PI_ARTICLE, RASPBERRY_PI_ANSWER],
]

# Shown instead of the full text, which would be unreadable in a table.
EXAMPLE_LABELS = [
    "Apollo 13 — the summary says the crew orbited the Moon",
    "Raspberry Pi — the answer holds up",
]
