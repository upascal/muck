Challenge Details

Dataset, submission, and evaluation
Everything you need to decide whether to compete

Register your team
The dataset
We’re providing a federal-government corpus centered on who is paying whom to shape policy, and how those efforts are described in public. It covers 2022 through March 2026 and combines two complementary record types:

Congressional lobbying filings from both chambers: registrants, clients, issue areas, and the staff moving between private firms and government offices.
Congressional press releases from the Congress Press corpus.
Registered teams receive the prepared corpus along with a manual describing its structure and contents.

What teams submit
We want teams to put in their best effort at uncovering genuinely newsworthy findings using AI agents. Dig into the provided corpus, bring in outside sources, pursue whatever seems interesting or important to you. Then — and this is the part that makes the competition matter beyond your own story — package up the workflows, prompts, and tools you built along the way into an Agent Skill that someone else could pick up and use on a different investigation.

An Agent Skill is a directory of instructions, scripts, and reference material that extends what a coding agent can do. The format is defined by the Agent Skills specification: a SKILL.md with YAML frontmatter and instructions, plus optional scripts/, references/, and assets/. Skills are more powerful than they look — if you haven’t built one before, these collections of journalism-oriented examples are a good starting point:

Florent Daudens — AI Journalism Skills
Joe Amditis — claude-skills-journalism
The skill you submit should capture things that would generalize past this particular investigation: a pattern for resolving messy entities, a review interface that makes provenance easy to check, a workflow that keeps a long-running investigation oriented across sessions. The specifics are up to you.

Each submission package contains four things:

Agent Skill(s). The primary artifact — the reusable workflow you built. You can bundle multiple skills that compose together. Skills should be self-contained and validate cleanly against the specification.
Findings report. A written summary of the newsworthy discoveries you produced by running the skill(s) on the corpus (and any outside material you brought in).
Interaction traces. Full logs of the model sessions, either as raw JSON or as a rendered page, including inputs, tool calls, outputs, and the moments where human judgment intervened. Traces should be keyed to the skill invocations they came from.
README.md. A brief map of the submission: included skills, which findings they support, where the relevant traces are located, any outside data used, any conflicts of interest, and whether findings suggest possible legal violations that should be flagged to the evaluation panel.
How we evaluate
A panel of journalists and technologists reviews each submission. There are two things they’re looking at: whether your findings hold up as journalism, and whether your skill is something another team could actually pick up and use.

Are the findings real?
Before anything else, panelists spot-check your findings report. Claims need to be accurate, sourced back to specific records, and of genuine public interest — undisclosed relationships, timing anomalies, spending patterns, that sort of thing. A polished summary of what the dataset contains isn’t a finding; something a reporter would chase is. Submissions that don’t clear this bar aren’t ranked further.

Is the skill useful?
Once findings clear that bar, the panel scores your skill on four dimensions. Panelists score each of the four technical dimensions independently on a 0-3 rubric, and the four dimensions are weighted equally by default. Reproducibility is assumed throughout: if an evaluator can’t re-run your skill, it cannot score above 1 on any dimension, regardless of how impressive it looks on paper.

Does it keep the investigation organized?
Investigations span days, sessions, and new leads that arrive mid-stream. A strong skill keeps track of what’s been checked, what’s still open, which entities matter, and which threads went cold, so the journalist doesn’t have to re-brief the agent every time they come back to it.

Is it efficient with the corpus?
Feeding thousands of documents through a model is expensive and slow. A strong skill pushes the heavy lifting — extraction, filtering, indexing, aggregation — to deterministic tools and cheaper models, so the main agent spends its budget on reasoning rather than scanning.

Can a human verify the work?
A reporter has to be able to stand behind what the agent produced. A strong skill ties every claim back to a specific source, presents evidence in a form that’s fast to review, and leaves a trace an editor could audit without re-doing the work.

Does it extend what an agent can do for investigations?
The most interesting skills provide real, novel capabilities: entity resolvers, cross-reference utilities, document-network traversal, domain-specific parsers. Tools that handle the messy, investigation-specific problems a generic coding agent can’t solve on its own, and that other teams could lift into their own work.

What gets published
Top submissions win cash prizes, get individual writeups on the GAIN blog, and may be invited to present their work at the Computation + Journalism Symposium. Any submission that clears the findings bar, validates against the Agent Skills specification, and shows real effort on all four dimensions above is included in a public, open source skills catalog.

Ethical considerations
Investigative findings may implicate real individuals and organizations. Teams should exercise the same editorial judgment they would apply in a newsroom context. If findings suggest potential legal violations, note that in the findings report and flag it to the evaluation panel. All participants agree to the code of conduct on registration.
