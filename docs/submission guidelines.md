Submissions are due July 15. Here's the submission form so you have it ready to go: https://docs.google.com/forms/d/e/1FAIpQLSfZi3Y22Q9xsaVN-TWg018vLf2HU7CKmRdf_2-9stiXNt0JUQ/viewform?usp=sharing&ouid=109507750837659658128
You’re welcome to bring in outside data, as long as it's not confidential or sensitive.
Evaluation criteria are on the Details page of the website: https://www.gain-agent-challenge.northwestern.edu/details/.



Agentic Investigation Challenge Submissions
Upload one `.zip` file containing all required submission materials. Name the file:

`[team-name]-agentic-investigation-submission.zip`

Your submission must include:

1. Agent Skill(s) - Include one or more complete Agent Skill directories. Each skill must include a `SKILL.md` file with YAML frontmatter and instructions. Optional `scripts/`, `references/`, and `assets/` directories should be included if your skill depends on them.

2. Findings report - Include a written report summarizing the newsworthy findings your team produced by running the skill(s) against the provided corpus. Each finding should be accurate, sourced to specific records, and explain its investigative relevance.

3. Interaction traces - Include full Claude Code session logs that produced the reported findings. These may be raw JSON/JSONL files or rendered transcript pages. Traces should show inputs, tool calls, outputs, and points where human judgment intervened. Please organize or label them so evaluators can connect each trace to the relevant skill invocation and finding.

Recommended package structure:

```
team-name-submission/
  README.md
  findings-report.md
  skills/
    skill-name/
      SKILL.md
      scripts/
      references/
      assets/
  traces/
    trace-01-skill-name.jsonl
    trace-02-skill-name.html
```

In `README.md`, briefly list the included skills, which findings they support, where the relevant traces are located, any outside data used, any conflicts of interest, and whether any findings suggest possible legal violations that should be flagged to the evaluation panel.

Claude Code transcript helper prompt

Teams can paste this into Claude Code from the directory where they are preparing their submission:

```
I need to package the Claude Code interaction traces relevant to this project for a competition submission.

Please help me find Claude Code transcript files for work done in this current project directory, then copy the relevant ones into a new ./traces directory. Do not delete, move, or modify the original transcript files.

Look for local Claude Code transcripts in likely locations such as ~/.claude/projects and any transcript path available from the current session. Identify candidate transcript files by checking whether they reference this current working directory, this project name, or the investigation/skill work in this repository.

For each candidate transcript, briefly summarize why it appears relevant. If the set is ambiguous, ask me before copying. If it is clear, copy the relevant transcript files into ./traces and rename them descriptively, preserving their original extension. Also create ./traces/README.md listing each copied transcript, its original path, and what finding or skill invocation it appears to support.
```
