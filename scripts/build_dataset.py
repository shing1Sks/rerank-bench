"""Author the benchmark dataset: 175 queries, 6 types, gold labels read from the built corpus.

Every gold chunk id was verified against data/chunks/chunks.jsonl while authoring
(see data/dataset/curation_notes.md). Negatives are verified topic-absent by the
companion check in scripts/check_negatives.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rerankbench.dataset import Query, validate_queries
from rerankbench.chunk import load_chunks
from rerankbench.config import load_config

Q = Query  # shorthand

# ---------------------------------------------------------------- verbatim (40)
VERBATIM = [
    Q("ver_01", 'which passage contains the line "Who-e debel you? - you no speak-e, dam-me, I kill-e"?', "verbatim", ["mobydick_c0040"], "mobydick", "quoted from p5 inn scene"),
    Q("ver_02", 'which passage contains the line "Supposing it be the captain of the Pequod, what dost thou want of him?"', "verbatim", ["mobydick_c0091"], "mobydick", "quoted from p11 tent exchange"),
    Q("ver_03", 'find the passage where the narrator calls the Pequod\'s crew Isolatoes, each living on a separate continent of his own', "verbatim", ["mobydick_c0144"], "mobydick", "p17 Isolatoes"),
    Q("ver_04", 'which passage lists the Folio whales beginning with the Sperm Whale, the Right Whale, and the Fin-Back Whale?', "verbatim", ["mobydick_c0160"], "mobydick", "p18 cetology folio list"),
    Q("ver_05", "which passage has Starbuck saying he came to hunt whales, not his commander's vengeance?", "verbatim", ["mobydick_c0192"], "mobydick", "p22 Starbuck objection"),
    Q("ver_06", "locate the reflection that chance plays within the right lines of necessity and is directed by free will", "verbatim", ["mobydick_c0248"], "mobydick", "p28 chance/necessity"),
    Q("ver_07", "which passage describes Ahab keeping his own private revenge against the man who stung him in the ventricles of his heart?", "verbatim", ["mobydick_c0296"], "mobydick", "p33 private revenge"),
    Q("ver_08", "find the passage suggesting Bishop Pontoppodan's great Kraken may ultimately resolve itself into the squid", "verbatim", ["mobydick_c0320"], "mobydick", "p36 kraken/squid"),
    Q("ver_09", "which passage compares the whale's sideways-placed eye to a young colt's eye?", "verbatim", ["mobydick_c0376"], "mobydick", "p42 lashless eye"),
    Q("ver_10", "locate the passage naming the whale and the sturgeon as the two royal fish of the English law writers", "verbatim", ["mobydick_c0456"], "mobydick", "p51 royal fish"),
    Q("ver_11", "which passage has Ahab telling Starbuck he is as true as the circumference to its centre?", "verbatim", ["mobydick_c0592"], "mobydick", "p66 circumference"),
    Q("ver_12", "find the passage stating that fundamental liberties protected by the Fourteenth Amendment's Due Process Clause extend to personal choices central to individual dignity and autonomy", "verbatim", ["scotus_c0002"], "scotus", "p1 due process holding"),
    Q("ver_13", "which passage states there is no difference between same- and opposite-sex couples with respect to this principle of marriage at the center of the legal and social order?", "verbatim", ["scotus_c0004"], "scotus", "p1 no difference"),
    Q("ver_14", "find the passage stating the first question presented: whether the Fourteenth Amendment requires a State to license a marriage between two people of the same sex", "verbatim", ["scotus_c0008"], "scotus", "p1 question presented"),
    Q("ver_15", "which passage cites Zablocki v. Redhail, which held the right to marry was burdened by a law about fathers behind on child support?", "verbatim", ["scotus_c0016"], "scotus", "p1 Zablocki cite"),
    Q("ver_16", "locate the passage about laws that search the sacred precincts of marital bedrooms and infringe the right to be let alone", "verbatim", ["scotus_c0052"], "scotus", "p1 marital bedrooms"),
    Q("ver_17", "which passage says the founding-era understanding of liberty was heavily influenced by John Locke and meant freedom from governmental action?", "verbatim", ["scotus_c0076"], "scotus", "p1 dissent Locke"),
    Q("ver_18", "find the dissent passage fearing those who cling to old beliefs may only whisper their thoughts in the recesses of their homes", "verbatim", ["scotus_c0092"], "scotus", "p1 whisper fears"),
    Q("ver_19", "which passage limits when the privilege of the writ of habeas corpus may be suspended?", "verbatim", ["constitution_c0006"], "constitution", "Art I Sec 9"),
    Q("ver_20", "locate the exact text of the oath to preserve, protect and defend the Constitution of the United States", "verbatim", ["constitution_c0009"], "constitution", "presidential oath"),
    Q("ver_21", "which passage defines treason as levying war against the United States or adhering to their enemies?", "verbatim", ["constitution_c0011"], "constitution", "Art III treason"),
    Q("ver_22", "find the clause stating no religious test shall ever be required as a qualification to any office or public trust", "verbatim", ["constitution_c0013"], "constitution", "Art VI religious test"),
    Q("ver_23", "which figure caption describes using a scanning tunneling microscope to see the individual atoms composing a sheet of gold?", "verbatim", ["openstax_phys_c0037"], "openstax_phys", "Fig 1.3 caption p20"),
    Q("ver_24", "find the caption explaining that an atomic clock uses the vibrations of cesium atoms to keep time to better than a microsecond per year", "verbatim", ["openstax_phys_c0050"], "openstax_phys", "Fig 1.8 caption p27"),
    Q("ver_25", "which passage uses a signpost giving distances and directions to towns to illustrate scalar and directional information?", "verbatim", ["openstax_phys_c0105"], "openstax_phys", "Fig 2.1 caption p53"),
    Q("ver_26", "locate the caption describing the corkscrew right-hand rule for determining the direction of a cross product", "verbatim", ["openstax_phys_c0174"], "openstax_phys", "Fig 2.30 caption p94"),
    Q("ver_27", "which caption shows a JR Central L0 series five-car maglev train undergoing a test run on the Yamanashi Test Track?", "verbatim", ["openstax_phys_c0212"], "openstax_phys", "Fig 3.1 caption p115"),
    Q("ver_28", "find the caption stating a hammer and a feather fall with the same constant acceleration if air resistance is negligible", "verbatim", ["openstax_phys_c0262"], "openstax_phys", "Fig 3.26 caption p147"),
    Q("ver_29", "which caption visualizes a function as an input/output device?", "verbatim", ["openstax_calc_c0026"], "openstax_calc", "Fig 1.2 caption p17"),
    Q("ver_30", "locate the caption stating that for any linear function the slope is independent of the choice of points on the line", "verbatim", ["openstax_calc_c0063"], "openstax_calc", "Fig 1.16 caption p45"),
    Q("ver_31", "which caption says the shape of a strand of silk in a spider's web can be described in terms of a hyperbolic function?", "verbatim", ["openstax_calc_c0151"], "openstax_calc", "Fig 1.49 caption p116"),
    Q("ver_32", "find the caption noting that for values of x close to 1 the graph of the square-root function and its tangent line appear to coincide", "verbatim", ["openstax_calc_c0292"], "openstax_calc", "Fig 3.5 caption p225"),
    Q("ver_33", "which page lists Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola as the authors?", "verbatim", ["d2l_c0000"], "d2l", "title page"),
    Q("ver_34", "find the section showing a 3-by-4 torch tensor and explaining indexing and slicing like Python lists", "verbatim", ["d2l_c0096"], "d2l", "p73 2.1.2"),
    Q("ver_35", "which passage introduces entropy as quantifying the amount of information contained in data?", "verbatim", ["d2l_c0240"], "d2l", "p131 entropy"),
    Q("ver_36", "locate the discussion that illustrates exploding gradients by multiplying 100 Gaussian random matrices with an initial matrix", "verbatim", ["d2l_c0324"], "d2l", "p226 exploding gradients"),
    Q("ver_37", "which passage states the reef system is composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres?", "verbatim", ["wiki_great_barrier_reef_c0000"], "wiki_great_barrier_reef", "lead paragraph"),
    Q("ver_38", "find the passage dating the revolution from the Estates General of 1789 to the Coup of 18 Brumaire on 9 November 1799", "verbatim", ["wiki_french_revolution_c0000"], "wiki_french_revolution", "lead paragraph"),
    Q("ver_39", "which passage gives the etymology of plate tectonics from the Ancient Greek for pertaining to building?", "verbatim", ["wiki_plate_tectonics_c0000"], "wiki_plate_tectonics", "lead etymology"),
    Q("ver_40", "locate the passage showing both the UK and US pronunciations of the word Renaissance", "verbatim", ["wiki_renaissance_c0000"], "wiki_renaissance", "lead pronunciation"),
]

# -------------------------------------------------------------- paraphrase (40)
PARAPHRASE = [
    Q("par_01", "where does the text describe the whaler's old weathered look, with masts originally lost overboard in a gale off Japan and replaced?", "paraphrase", ["mobydick_c0089"], "mobydick", "p10 venerable bows"),
    Q("par_02", "what passage admits Queequeg held strange beliefs about a small idol and his private devotions yet seemed to know what he was about?", "paraphrase", ["mobydick_c0104"], "mobydick", "p12 Yojo/Ramadan"),
    Q("par_03", "find where Ahab spends an evening alone at the bulwark smoking his pipe after Stubb has gone below", "paraphrase", ["mobydick_c0152"], "mobydick", "p17 ch 30 The Pipe"),
    Q("par_04", "which passage depicts solemn silent dinners at the captain's table, with a rat's racket below decks a relief to Stubb?", "paraphrase", ["mobydick_c0176"], "mobydick", "p20 cabin meals"),
    Q("par_05", "where does the text say that pointing a lance at the sperm whale would be an inevitable death-warrant for any mortal man?", "paraphrase", ["mobydick_c0208"], "mobydick", "p24 daring warfare"),
    Q("par_06", "which discussion argues that whiteness lends a special virtue and even royal preeminence to things like marble, flowers, and pearls?", "paraphrase", ["mobydick_c0216"], "mobydick", "p25 whiteness"),
    Q("par_07", "find the joke about using a wooden leg to stop a boat's plug-hole instead of going out in a boat", "paraphrase", ["mobydick_c0264"], "mobydick", "p30 timber toe"),
    Q("par_08", "where is the confusion between ambergris and grey amber described as a long-standing puzzle for the learned?", "paraphrase", ["mobydick_c0464"], "mobydick", "p52 ambergris origin"),
    Q("par_09", "which passage traces how attitudes toward gay and lesbian people moved from mid-20th-century condemnation into political and judicial view?", "paraphrase", ["scotus_c0012"], "scotus", "p1 historical shift"),
    Q("par_10", "find the passage pointing to children raised by same-sex parents as powerful confirmation that gays and lesbians create loving, supportive families", "paraphrase", ["scotus_c0020"], "scotus", "p1 same-sex parents"),
    Q("par_11", "where does the opinion describe the disruption when a spouse cannot visit a hospitalized partner because a neighboring state will not recognize the marriage?", "paraphrase", ["scotus_c0032"], "scotus", "p1 hospitalization harm"),
    Q("par_12", "which passage recounts states responding to a 2003 state-court marriage ruling by enacting constitutional amendments defining marriage traditionally?", "paraphrase", ["scotus_c0044"], "scotus", "p1 state amendments"),
    Q("par_13", "locate the criticism, quoting Learned Hand, that the Lochner-era court amounted to a legislative chamber elevating judges' policy judgments", "paraphrase", ["scotus_c0048"], "scotus", "p1 Lochner critique"),
    Q("par_14", "where does the dissent say the Constitution says nothing about same-sex marriage and leaves the question to the people of each state?", "paraphrase", ["scotus_c0088"], "scotus", "p1 leaves to states"),
    Q("par_15", "what text caps the size of the House at no more than one representative for every thirty thousand people?", "paraphrase", ["constitution_c0001"], "constitution", "Art I Sec 2"),
    Q("par_16", "which passage governs how a bill passed over a veto must record the yeas and nays and the names of members voting for and against?", "paraphrase", ["constitution_c0004"], "constitution", "Art I Sec 7"),
    Q("par_17", "where is Congress given the power to create lower federal courts and to define and punish piracies committed on the high seas?", "paraphrase", ["constitution_c0005"], "constitution", "Art I Sec 8"),
    Q("par_18", "find the section forbidding individual states from keeping troops or ships of war in time of peace without consent of Congress", "paraphrase", ["constitution_c0007"], "constitution", "Art I Sec 10"),
    Q("par_19", "which passage lists geological work that relies heavily on physics, such as radioactive dating of rocks and earthquake analysis?", "paraphrase", ["openstax_phys_c0036"], "openstax_phys", "p19 geology links"),
    Q("par_20", "where is a table laying out SI base quantities like electrical current, thermodynamic temperature, and luminous intensity?", "paraphrase", ["openstax_phys_c0048"], "openstax_phys", "Table 1.1 p26"),
    Q("par_21", "find the general formula expressing the dimension of any physical quantity as powers of length, mass, and time", "paraphrase", ["openstax_phys_c0060"], "openstax_phys", "p32 dimensions"),
    Q("par_22", "which section teaches how to determine the correct number of significant figures for the result of a computation?", "paraphrase", ["openstax_phys_c0072"], "openstax_phys", "p38 1.6 objectives"),
    Q("par_23", "where does the text define the accuracy of a measurement as how close it is to an accepted reference value?", "paraphrase", ["openstax_phys_c0090"], "openstax_phys", "p47 accuracy"),
    Q("par_24", "locate the worked example asking how far and in what direction a skier must travel from a rest point to return directly to the lodge", "paraphrase", ["openstax_phys_c0156"], "openstax_phys", "p84 skier return"),
    Q("par_25", "which passage notes that the scalar multiplication of two vectors is also called the dot product?", "paraphrase", ["openstax_phys_c0186"], "openstax_phys", "p101 glossary note"),
    Q("par_26", "find the formal definition of a function consisting of a set of inputs, a set of outputs, and a rule assigning each input to one output", "paraphrase", ["openstax_calc_c0025"], "openstax_calc", "p16 definition"),
    Q("par_27", "which example shows that the reciprocal of a polynomial that is never zero has all real numbers as its domain?", "paraphrase", ["openstax_calc_c0045"], "openstax_calc", "p31 g(f(x)) domain"),
    Q("par_28", "where is the point-slope equation of a linear function introduced as f(x) minus a value equals m times x minus a point?", "paraphrase", ["openstax_calc_c0065"], "openstax_calc", "p46 point-slope"),
    Q("par_29", "which passage explains that adding a constant shifts a graph upward and subtracting one shifts it downward?", "paraphrase", ["openstax_calc_c0085"], "openstax_calc", "p61 vertical shifts"),
    Q("par_30", "find where inverse functions are shown to undo each other in both directions, f inverse of f of x equals x", "paraphrase", ["openstax_calc_c0115"], "openstax_calc", "p86 inverse identity"),
    Q("par_31", "which theorem says functions continuous over a closed interval take every value between the values at its endpoints?", "paraphrase", ["openstax_calc_c0250"], "openstax_calc", "p196 IVT"),
    Q("par_32", "where does the book walk through setting up a Python working environment by installing Miniconda?", "paraphrase", ["d2l_c0036"], "d2l", "p34 installation"),
    Q("par_33", "which passage describes the usual training process starting from a randomly initialized model that cannot yet deliver the desired behavior?", "paraphrase", ["d2l_c0048"], "d2l", "p44 training loop"),
    Q("par_34", "find the section introducing derivatives of functions of many variables, called partial derivatives and gradients, for deep learning", "paraphrase", ["d2l_c0126"], "d2l", "p98 partials"),
    Q("par_35", "where is independence among successive draws from a distribution explained as the thing that enables strong statistical conclusions?", "paraphrase", ["d2l_c0150"], "d2l", "p112 independence"),
    Q("par_36", "which code block defines a multilayer perceptron class with configurable hidden width using flatten, a lazy linear layer, and ReLU?", "paraphrase", ["d2l_c0312"], "d2l", "p218 MLP class"),
    Q("par_37", "which passage splits photosynthesis into a light-dependent stage producing NADPH and ATP and a light-independent stage that builds sugars?", "paraphrase", ["wiki_photosynthesis_c0003"], "wiki_photosynthesis", "two stages"),
    Q("par_38", "where is quantum mechanics characterized as the fundamental theory of matter and light at and below the scale of atoms?", "paraphrase", ["wiki_quantum_mechanics_c0000"], "wiki_quantum_mechanics", "lead"),
    Q("par_39", "find the passage about the war that began with Austria and Prussia and later drew in Spain, Portugal, Naples, and Tuscany", "paraphrase", ["wiki_french_revolution_c0015"], "wiki_french_revolution", "First Coalition"),
    Q("par_40", "which passage notes the reef's green sea turtles have two genetically distinct populations and seagrass beds attract dugongs?", "paraphrase", ["wiki_great_barrier_reef_c0006"], "wiki_great_barrier_reef", "turtles/dugongs"),
]

# --------------------------------------------------------------- reasoning (30)
REASONING = [
    Q("rea_01", "why does the narrator hesitate at the tent door before shipping on the Pequod, and what do the part-owner and Quaker seller tell him about whaling pay and lay before he commits?", "reasoning", ["mobydick_c0091", "mobydick_c0092"], "mobydick", "p11 negotiation pair"),
    Q("rea_02", "how does Queequeg's claimed skill at picking the best whaler hold up when the narrator finally inspects the ship he chose?", "reasoning", ["mobydick_c0088", "mobydick_c0089"], "mobydick", "p10 sagacity + description"),
    Q("rea_03", "which two passages show the first mate's objection that vengeance yields no barrels, and the captain's concealed private revenge that answers it?", "reasoning", ["mobydick_c0192", "mobydick_c0296"], "mobydick", "mate vs captain motive"),
    Q("rea_04", "which two passages capture the landlord's prank of pairing the narrator with a harpooneer he has never met, and the frightening first meeting that follows?", "reasoning", ["mobydick_c0032", "mobydick_c0040"], "mobydick", "inn prank + meeting"),
    Q("rea_05", "how does the Court answer the respondents' claim that extending marriage to same-sex couples would demean a timeless institution?", "reasoning", ["scotus_c0001", "scotus_c0002"], "scotus", "respondents vs holding"),
    Q("rea_06", "which passages supply a legal precedent on the burdened right to marry and modern family evidence that together support marriage as foundational?", "reasoning", ["scotus_c0016", "scotus_c0020"], "scotus", "precedent + families"),
    Q("rea_07", "which passages voice the dissent's worry about unelected judges substituting their own policy judgments for democratic choices?", "reasoning", ["scotus_c0048", "scotus_c0060"], "scotus", "dissent institutional"),
    Q("rea_08", "how does the dissent's historical account of liberty as freedom from government support its claim that the Constitution leaves marriage to the states?", "reasoning", ["scotus_c0076", "scotus_c0088"], "scotus", "liberty history chain"),
    Q("rea_09", "which passages grant Congress its war, court, and piracy powers, and then the limits on them such as habeas corpus and bills of attainder?", "reasoning", ["constitution_c0005", "constitution_c0006"], "constitution", "powers vs limits"),
    Q("rea_10", "what constitutional text sets out the president's oath and the take-care duty, and what standard for removal of civil officers follows them?", "reasoning", ["constitution_c0009", "constitution_c0010"], "constitution", "oath + execution"),
    Q("rea_11", "which passages define treason narrowly and then bar religious tests for office, both constraints on state power?", "reasoning", ["constitution_c0011", "constitution_c0013"], "constitution", "treason + no test"),
    Q("rea_12", "which table lists the SI base quantities, and which worked step shows how to restate a kilogram-prefixed value as a number between one and a thousand?", "reasoning", ["openstax_phys_c0048", "openstax_phys_c0054"], "openstax_phys", "table + prefix step"),
    Q("rea_13", "how is multiplying a vector by a scalar illustrated with a doubled length, and how is vector subtraction related to adding a reversed vector?", "reasoning", ["openstax_phys_c0114", "openstax_phys_c0117"], "openstax_phys", "scalar mult + difference"),
    Q("rea_14", "which passages tie a vector's magnitude to its components by the Pythagorean theorem and its direction angle to the components by a cosine?", "reasoning", ["openstax_phys_c0132", "openstax_phys_c0133"], "openstax_phys", "magnitude + angle"),
    Q("rea_15", "what do the tangent-line figure for position-time graphs and the paired position/velocity figure together teach about reading velocity off a graph?", "reasoning", ["openstax_phys_c0221", "openstax_phys_c0228"], "openstax_phys", "Fig 3.6 + Fig 3.9"),
    Q("rea_16", "which two figures explain why a particle in circular motion accelerates toward the center, and how a tangential acceleration combines with it?", "reasoning", ["openstax_phys_c0336", "openstax_phys_c0342"], "openstax_phys", "Fig 4.18 + Fig 4.23"),
    Q("rea_17", "how do the hockey-puck example and the air-table demonstration together establish Newton's first law about objects at rest and in motion?", "reasoning", ["openstax_phys_c0385", "openstax_phys_c0387"], "openstax_phys", "first law pair"),
    Q("rea_18", "which passages define the slope of a secant line and then define instantaneous velocity as the limit that average velocities approach?", "reasoning", ["openstax_calc_c0177", "openstax_calc_c0180"], "openstax_calc", "secant then limit"),
    Q("rea_19", "how do the two worked limits, one near x equals 2 and one for a quotient with a vanishing denominator, both use graphs to confirm table estimates?", "reasoning", ["openstax_calc_c0190", "openstax_calc_c0195"], "openstax_calc", "Fig 2.12 + Fig 2.14"),
    Q("rea_20", "which passages state the squeeze theorem and then apply it to the sine function squeezed on the unit circle to get a limit at zero?", "reasoning", ["openstax_calc_c0230", "openstax_calc_c0231"], "openstax_calc", "theorem + application"),
    Q("rea_21", "which figures and example together classify the discontinuity of a rational function at the point where its formula has a hole?", "reasoning", ["openstax_calc_c0240", "openstax_calc_c0245"], "openstax_calc", "Fig 2.33 + Ex 2.30"),
    Q("rea_22", "which passages give the secant-line difference quotient and then rewrite the derivative as the limit of the ratio of small changes dy over dx?", "reasoning", ["openstax_calc_c0290", "openstax_calc_c0312"], "openstax_calc", "msec then dy/dx"),
    Q("rea_23", "which passages contrast programs written as rigid rule sets with models learned from data, and then describe training from random initialization?", "reasoning", ["d2l_c0042", "d2l_c0048"], "d2l", "rules vs learned"),
    Q("rea_24", "which passages set the frequentist axiom that an event and its complement sum to one against the Bayesian view of degrees of belief?", "reasoning", ["d2l_c0138", "d2l_c0144"], "d2l", "bayes vs axioms"),
    Q("rea_25", "which passages give the compact linear model as a dot product of weights and features, and then penalize the squared L2 norm of those weights?", "reasoning", ["d2l_c0168", "d2l_c0222"], "d2l", "model then penalty"),
    Q("rea_26", "which two anecdotes show a classifier latching onto spurious cues - loan decisions tied to shoe style, and trees distinguished by shadows?", "reasoning", ["d2l_c0276", "d2l_c0282"], "d2l", "spurious correlations"),
    Q("rea_27", "which passages explain why raw class scores need squashing into probabilities, and then define the entropy concept used to score those probabilities?", "reasoning", ["d2l_c0234", "d2l_c0240"], "d2l", "softmax then entropy"),
    Q("rea_28", "which passages describe entanglement's uses in computing and communication, and its role when a measuring apparatus joins the system being measured?", "reasoning", ["wiki_quantum_mechanics_c0003", "wiki_quantum_mechanics_c0009"], "wiki_quantum_mechanics", "uses + measurement"),
    Q("rea_29", "which passages trace excited electrons along a chain of acceptors, and then name the enzyme that captures carbon dioxide during sugar building?", "reasoning", ["wiki_photosynthesis_c0006", "wiki_photosynthesis_c0021"], "wiki_photosynthesis", "electron chain + RuBisCO"),
    Q("rea_30", "which passages credit Wegener's drifting Pangaea proposal, and then list the earlier thinkers who had already noticed the continents fit together?", "reasoning", ["wiki_plate_tectonics_c0012", "wiki_plate_tectonics_c0015"], "wiki_plate_tectonics", "Wegener + predecessors"),
]

# --------------------------------------------------------------- negative (20)
NEGATIVE = [
    Q("neg_01", "which amendment prohibits quartering soldiers in private homes during peacetime?", "negative", [], "none", "Third Amendment not in our constitution transcription chunks (verified)"),
    Q("neg_02", "2024 Formula 1 championship standings and driver points totals", "negative", [], "none", "motorsport absent"),
    Q("neg_03", "how to replace the water filter inside a Whirlpool refrigerator", "negative", [], "none", "appliance repair absent"),
    Q("neg_04", "the cricket rule for a leg before wicket dismissal explained", "negative", [], "none", "sport rules absent"),
    Q("neg_05", "steps for brewing kombucha safely at home with a scoby", "negative", [], "none", "food prep absent"),
    Q("neg_06", "current population of the Tokyo metropolitan area", "negative", [], "none", "demographics absent"),
    Q("neg_07", "how Python's asyncio event loop schedules coroutines", "negative", [], "none", "d2l covers ML, not asyncio (verified)"),
    Q("neg_08", "the plot twist structure of the movie Inception", "negative", [], "none", "film absent"),
    Q("neg_09", "how to fill out IRS form 1040 schedule C for self-employment", "negative", [], "none", "tax procedure absent"),
    Q("neg_10", "chord progressions typically used in jazz improvisation over ii-V-I", "negative", [], "none", "music theory absent"),
    Q("neg_11", "symptoms and recommended treatment for tennis elbow", "negative", [], "none", "medicine absent"),
    Q("neg_12", "oven temperature and hydration ratios for sourdough bread", "negative", [], "none", "baking absent"),
    Q("neg_13", "React useEffect cleanup function behavior explained", "negative", [], "none", "web framework absent"),
    Q("neg_14", "Bitcoin's proof-of-work difficulty adjustment schedule", "negative", [], "none", "cryptocurrency absent"),
    Q("neg_15", "construction status of the Nicaraguan Canal project", "negative", [], "none", "infrastructure news absent"),
    Q("neg_16", "how to train a golden retriever puppy to sit and stay", "negative", [], "none", "pet training absent"),
    Q("neg_17", "summary of Tennessee Williams' play The Glass Menagerie", "negative", [], "none", "drama absent"),
    Q("neg_18", "the scoring rules for Olympic figure skating programs", "negative", [], "none", "phys has skater examples, not skating rules (near-miss negative)"),
    Q("neg_19", "a cheat sheet pairing camera aperture with shutter speed settings", "negative", [], "none", "photography absent"),
    Q("neg_20", "the order of battle of the fleets at Trafalgar", "negative", [], "none", "naval history absent from corpus (verified)"),
]

# -------------------------------------------------------------- structural (20)
STRUCTURAL = [
    Q("str_01", "in which chapter and on what page does the book's treatment of the Transformer architecture begin?", "structural", ["d2l_c0012"], "d2l", "TOC entry 11.7 p440"),
    Q("str_02", "where in the table of contents is the Builders' Guide chapter and its Layers and Modules section located?", "structural", ["d2l_c0006"], "d2l", "TOC ch6 p207"),
    Q("str_03", "which page does the section on word embedding with global vectors, GloVe, start on according to the contents?", "structural", ["d2l_c0018"], "d2l", "TOC 15.5 p711"),
    Q("str_04", "on what page does the chapter on vectors begin in this physics volume's contents listing?", "structural", ["openstax_phys_c0006"], "openstax_phys", "TOC ch2 p43"),
    Q("str_05", "find the contents entries that give page numbers for center of mass and rocket propulsion", "structural", ["openstax_phys_c0012"], "openstax_phys", "TOC 9.6/9.7"),
    Q("str_06", "where does the discussion of sound intensity appear according to the book's contents pages?", "structural", ["openstax_phys_c0018"], "openstax_phys", "TOC 17.3 p859"),
    Q("str_07", "which unit and chapter listing shows where Newton's laws of motion are covered in this volume?", "structural", ["openstax_phys_c0024"], "openstax_phys", "volume outline p12"),
    Q("str_08", "what page does the antiderivatives section start on, and which chapter on integration follows it?", "structural", ["openstax_calc_c0010"], "openstax_calc", "TOC 4.10 p485"),
    Q("str_09", "which front-matter passage lists the novel's chapter titles such as Loomings and The Spouter-Inn?", "structural", ["mobydick_c0000"], "mobydick", "contents p1"),
    Q("str_10", "which passage identifies this document as the National Archives transcription of the Constitution?", "structural", ["constitution_c0000"], "constitution", "page header note"),
    Q("str_11", "in which passage covering the Senate's officers does the President pro tempore get chosen?", "structural", ["constitution_c0002"], "constitution", "Art I Sec 3"),
    Q("str_12", "where does the Constitution describe how votes are taken by states when choosing the president in the House?", "structural", ["constitution_c0008"], "constitution", "Art II Sec 1"),
    Q("str_13", "which passage covers Congress's power to make rules for United States territory and other property?", "structural", ["constitution_c0012"], "constitution", "Art IV Sec 3"),
    Q("str_14", "which passage names the case and notes the lower court's judgment was reversed?", "structural", ["scotus_c0000"], "scotus", "syllabus header"),
    Q("str_15", "where does the syllabus frame the two questions the Court limited its review to?", "structural", ["scotus_c0008"], "scotus", "questions presented"),
    Q("str_16", "which passage lists the state lower-court decisions with their reporter citations, including Cote-Whitacre and Lewis v. Harris?", "structural", ["scotus_c0036"], "scotus", "citation list"),
    Q("str_17", "which passage marks the start of the Hungary section with a subheading right after Emperor Maximilian?", "structural", ["wiki_renaissance_c0021"], "wiki_renaissance", "=== Hungary === heading"),
    Q("str_18", "where does the article's fishing section begin?", "structural", ["wiki_great_barrier_reef_c0021"], "wiki_great_barrier_reef", "=== Fishing === heading"),
    Q("str_19", "which passage heads the section on the political crisis and the fall of the Girondins?", "structural", ["wiki_french_revolution_c0015"], "wiki_french_revolution", "section heading"),
    Q("str_20", "which short repeated line states where this OpenStax book is available for free?", "structural", ["openstax_phys_c0030"], "openstax_phys", "cnx.org footer line"),
]

# --------------------------------------------------------------- graphical (25)
GRAPHICAL = [
    Q("gra_01", "which figure deliberately shows an image that could be a whirlpool in a tank of water or a collage of paint and beads?", "graphical", ["openstax_phys_c0031"], "openstax_phys", "Fig 1.1 p17"),
    Q("gra_02", "which figure is the table showing the orders of magnitude of length, mass, and time?", "graphical", ["openstax_phys_c0041"], "openstax_phys", "Fig 1.4 p22"),
    Q("gra_03", "which figure shows an atomic clock that keeps time using the vibrations of cesium atoms?", "graphical", ["openstax_phys_c0050"], "openstax_phys", "Fig 1.8 p27"),
    Q("gra_04", "find the figure about redefining the SI unit of mass with complementary methods under investigation", "graphical", ["openstax_phys_c0052"], "openstax_phys", "Fig 1.10 p28"),
    Q("gra_05", "which figure draws a 6 kilometre displacement to scale as a 12 centimetre vector?", "graphical", ["openstax_phys_c0110"], "openstax_phys", "Fig 2.4 p56"),
    Q("gra_06", "which figure illustrates the various possible relations between two vectors, including unequal and anti-parallel cases?", "graphical", ["openstax_phys_c0111"], "openstax_phys", "Fig 2.5 p57"),
    Q("gra_07", "which figure applies the parallelogram rule four times to produce a green resultant vector for a trip through several cities?", "graphical", ["openstax_phys_c0124"], "openstax_phys", "Fig 2.11 p64"),
    Q("gra_08", "which figure demonstrates the corkscrew right-hand rule for finding the direction of a cross product?", "graphical", ["openstax_phys_c0174"], "openstax_phys", "Fig 2.30 p94"),
    Q("gra_09", "which chapter-opener photo shows a five-car maglev train on a test track?", "graphical", ["openstax_phys_c0212"], "openstax_phys", "Fig 3.1 p115"),
    Q("gra_10", "which figure graphs a lecturer's position versus time and marks the average velocity as a connecting line's slope?", "graphical", ["openstax_phys_c0219"], "openstax_phys", "Fig 3.5 p120"),
    Q("gra_11", "which figure shows an object moving east that decelerates, stops, and reverses, passing back through its origin?", "graphical", ["openstax_phys_c0232"], "openstax_phys", "Fig 3.11 p128"),
    Q("gra_12", "which figure traces a particle's random Brownian displacements with the total displacement drawn in red?", "graphical", ["openstax_phys_c0305"], "openstax_phys", "Fig 4.6 p172"),
    Q("gra_13", "which figure shows the tangential and centripetal acceleration vectors and their sum as the net acceleration?", "graphical", ["openstax_phys_c0342"], "openstax_phys", "Fig 4.23 p199"),
    Q("gra_14", "which figure walks through the steps of drawing a free-body diagram with normal force, weight, and friction labeled?", "graphical", ["openstax_phys_c0380"], "openstax_phys", "Fig 5.4 p220"),
    Q("gra_15", "which figure pairs a mountain climber pulling down on a rope with the rope's equal pull up on the climber?", "graphical", ["openstax_phys_c0417"], "openstax_phys", "Fig 5.17 p241"),
    Q("gra_16", "which chapter-opener photograph shows a portion of the San Andreas Fault in California?", "graphical", ["openstax_calc_c0023"], "openstax_calc", "Fig 1.1 p15"),
    Q("gra_17", "which figure visualizes a function as an input-output machine?", "graphical", ["openstax_calc_c0026"], "openstax_calc", "Fig 1.2 p17"),
    Q("gra_18", "which figure contrasts a graph symmetric about the y-axis with a graph symmetric about the origin?", "graphical", ["openstax_calc_c0050"], "openstax_calc", "Fig 1.13 p34"),
    Q("gra_19", "which figure shows that a line's slope is the same no matter which two points on the line you choose?", "graphical", ["openstax_calc_c0063"], "openstax_calc", "Fig 1.16 p45"),
    Q("gra_20", "which figure illustrates shifting a graph left or right by a constant number of units?", "graphical", ["openstax_calc_c0086"], "openstax_calc", "Fig 1.24 p62"),
    Q("gra_21", "which figure defines radian measure through the arc length an angle cuts from the unit circle?", "graphical", ["openstax_calc_c0096"], "openstax_calc", "Fig 1.30 p70"),
    Q("gra_22", "which figure states the horizontal line test with one graph that passes and one that fails?", "graphical", ["openstax_calc_c0116"], "openstax_calc", "Fig 1.38 p87"),
    Q("gra_23", "which figure poses the area problem as finding the area of a shaded region under a curve?", "graphical", ["openstax_calc_c0181"], "openstax_calc", "Fig 2.8 p137"),
    Q("gra_24", "which figure uses the graph of one over x to show a limit that does not exist at zero?", "graphical", ["openstax_calc_c0205"], "openstax_calc", "Fig 2.19 p156"),
    Q("gra_25", "which figure shows the absolute value function as continuous at zero yet not differentiable there?", "graphical", ["openstax_calc_c0316"], "openstax_calc", "Fig 3.14 p246"),
]

ALL = VERBATIM + PARAPHRASE + REASONING + NEGATIVE + STRUCTURAL + GRAPHICAL


def main() -> int:
    cfg = load_config()
    chunks = load_chunks(cfg.data_dir / "chunks" / "chunks.jsonl")
    chunk_ids = {c.id for c in chunks}

    errs = validate_queries(ALL, chunk_ids)
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs:
            print(" ", e)
        return 1

    out = cfg.data_dir / "dataset" / "queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in ALL:
            f.write(json.dumps(q.__dict__, ensure_ascii=False) + "\n")
    from collections import Counter
    by_type = Counter(q.query_type for q in ALL)
    by_doc = Counter(q.source_doc for q in ALL)
    print(f"wrote {len(ALL)} queries -> {out}")
    print("by type:", dict(by_type))
    print("by source:", dict(sorted(by_doc.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
