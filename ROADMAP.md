# Project Roadmap (Plain-Language Version)

**In one paragraph:** This project builds two small tools that work together to catch coal power plants "in the act" of emitting CO2, using only satellite data. One tool looks at pictures from space and guesses "is this a power plant or not?" The other tool measures a CO2 bump in the air over a plant and turns that into a "tons of CO2 per year" number. Both tools started rough, got tested honestly (which revealed they were doing worse than first thought), got fixed, and — most importantly — got checked against India's real, government-reported emissions data for the first time. That check is now done, and it shows the tools are in the right ballpark, with a small working fix that already makes them more accurate.

---

## 1. What is this project?

Coal power plants are one of the biggest sources of CO2 in the world. In India, there isn't an easy way for an outsider to independently check how much CO2 a specific power plant is actually emitting — you mostly have to trust what's reported. This project tries to build that independent check, using free satellite data and some physics/math, not the plant's own paperwork.

It has two halves that work side-by-side but aren't yet fully connected:

## 2. The two tracks, explained simply

**Track A — "the Spotter."** A small AI model that looks at a 64×64-pixel satellite picture and decides: "power plant" or "not a power plant." It learns from pictures of gases (NO2, SO2) and heat (VIIRS) that power plants tend to give off. To make sure it's not cheating, it's also shown lots of tricky "not a power plant" pictures — cities, steel factories, highways — things that also produce pollution but aren't power plants.

**Track B — "the Calculator."** A physics-based method that looks at satellite measurements of CO2 in the air near a plant, compares that to the CO2 level a bit farther away (the "background"), and does the math to estimate how many tons of CO2 per year the plant is emitting. It also accounts for wind (since wind blows the CO2 around) and gives an honest "plus or minus" uncertainty range, not just a single number.

## 3. The story so far (in plain words)

1. **Building the Spotter (Weeks 2–5).** Started simple: one satellite gas channel (NO2), 91% accurate. But it turned out the model was just spotting "anything burning a lot of fuel," not power plants specifically — it kept mistaking cities and steel plants for power plants. Adding more data channels (SO2, heat) and harder practice examples slowly narrowed this down, though accuracy actually went *down* at first (from 91% to 77%) as the harder, more honest test set exposed the problem.

2. **Building the Calculator (Weeks 6–10).** Started estimating CO2 for a handful of plants. Found and fixed a real math bug in how wind speed was averaged (it was accidentally cancelling itself out over a year). Added a proper "how confident are we" uncertainty number, made of three parts: wind uncertainty, measurement noise, and how sensitive the answer is to exactly where you draw the "background" boundary.

3. **Catching the Spotter cheating (Week 11).** Discovered the Spotter's test was flawed — it was being tested on some of the same plants (just different months) it trained on, like a student seeing exam questions in advance. Fixing this dropped the honest accuracy a lot, revealing the model wasn't as good as it looked.

4. **Growing from 4 plants to 20.** More real-world examples were added so the Spotter had more to learn from, and the Calculator had more plants to estimate.

5. **The generalization scare, and the real fix.** After growing to 20 plants, a single test showed 88% accuracy at recognizing plants it hadn't specifically trained on — great news, or so it seemed. But a much more thorough test (holding out *each* plant, one at a time, and checking every single one) showed the true number was only **47%** — barely better than a coin flip. The 88% had just gotten lucky with which plants were tested. Two fixes were tried: giving the model more example "practice" through simple image tricks (flipping/rotating pictures) — this didn't help at all — and giving it real additional satellite observations (a second year of data) — this genuinely worked, pushing the honest accuracy up to **69%**. Two specific plants still can't be recognized reliably, but that's now understood to be because their actual pollution signal is too faint for the satellite to see clearly — not a fixable bug.

6. **Solving three "impossible" readings.** Three plants showed a CO2 reading *lower* than the background air around them — physically strange for something that's supposed to be emitting CO2. Careful detective work found the real cause each time: either the reading was just noisy and not really different from zero, or (in one tricky case) the "near the plant" measurements happened to all come from a cold month while the "background" measurements were blended from warmer months — comparing apples to oranges. Fixing the comparison to use only matching months made the numbers make sense again.

7. **The big breakthrough: finding real ground truth.** For most of the project, there was no way to check the Calculator's answers against reality — only against another satellite-based estimate (Climate TRACE) that has its own unknown accuracy for India. This changed when India's official government electricity authority (CEA) was found to publish a public database of real, reported power-plant emissions — based on how much fuel each plant actually burned, not satellite guessing. All of this project's power plants were found in that database by name. For the first time, the Calculator's numbers could be checked against something real. They were found to be off by roughly 2.5–3× on average — but a simple correction, based on how noisy the background readings were for a given plant, was built and tested honestly (never "peeking" at the answer while training). It measurably improved the accuracy.

8. **Tidying up.** Added a small automatic test suite so future changes to the core math can be checked in under a second instead of by eye. Found and fixed a leftover math bug (dividing by zero in a rare case). Resolved one more plant's odd negative reading using the same detective method as step 6.

## 4. Where things stand today

| Question | Simple answer |
|---|---|
| How many power plants does this cover? | 21 fully processed (20 original + 1 more), plus 9 more identified but not yet processed |
| How good is the Spotter at recognizing a brand-new plant? | **69%** (up from an honest 47%, after real fixes) — meaning it correctly recognizes about 7 out of 10 unseen plants; 2 specific plants are known dead ends (too weak a signal for satellites to catch) |
| How accurate is the Calculator's CO2 estimate? | Roughly a factor of ~2.5× off on average compared to real government data — with a working fix that's already made it noticeably better, but it's still an early-stage estimate, not something to bet money on yet |
| Is any of this tested automatically? | Yes — a small test suite checks the core math every time it's run, though it has to be run by hand (no auto-run-on-every-change setup yet) |
| Are the two tracks (Spotter + Calculator) combined into one system? | Not yet — they still work side-by-side, not together |

## 5. What's left to do — step by step

1. **Decide what to do with the 9 unprocessed candidate plants.** They're on the "maybe later" list right now. Either process them properly (adding more real-world coverage) or formally decide not to and move on — leaving it undecided indefinitely isn't ideal.
2. ~~Explain the two Spotter dead-end plants~~ — already done: they're confirmed to be a real satellite-sensitivity limit, not a bug, so no more work needed there.
3. **Make the ground-truth correction sturdier.** Right now it's based on only one factor and one year of real government data, checked against 17 plants. Next steps: try combining a second helpful factor, pull a few more years of the real government data to make the fix more reliable, and extend it to cover plants that currently fall outside the check.
4. **Set up automatic testing.** The test suite exists but someone has to remember to run it. Wiring it up to run automatically whenever code changes would catch mistakes faster.
5. **Explain the ~1-month gap in project history** (mid-July to mid-August) where nothing was recorded — minor, just for completeness, not urgent.
6. **Long-term goal: actually combine the Spotter and the Calculator into one system.** Right now they're two separate tools. The eventual goal is a single system where the Spotter's "is this really a power plant, and how confident am I?" feeds directly into the Calculator's CO2 estimate, making the whole thing more trustworthy end-to-end. This is the biggest remaining piece of work and isn't blocked on anything except time and further experimentation.
7. **Eventually: a clean, shareable write-up.** Once the above settles, package the findings (the honest wins, the honest failures, and the real-ground-truth validation) into a clear final report — the `RESEARCH_PAPER.md` file already exists as a working draft of this.

---

*This file is a plain-language companion to the more detailed project documents: `README.md` (setup + current status), `NEXT_STEPS.md` (running technical log), `RESEARCH_PAPER.md` (full write-up), and `PROJECT_RESEARCH_DOCUMENTATION.md` (exhaustive research history). Read those for the full technical detail behind every claim above.*
