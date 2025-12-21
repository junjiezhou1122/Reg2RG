---
name: paper-reading
description: Systematic academic paper analysis with structured breakdowns, critical evaluation, and research connections. Use when user mentions reading, analyzing, or understanding research papers.
---

# Paper Reading & Analysis Skill

**When to use:** User mentions analyzing a paper, understanding research, or provides a PDF/arXiv link/paper content.

## Core Philosophy

**Deep reading over skimming.** Extract the problem, solution, evidence, and implications methodically. Connect to broader research context. Identify both strengths and limitations critically.

---

## Paper Analysis Workflow

### 1. **Initial Scan** (Get Oriented)
- Title, authors, venue, year
- Abstract summary in one sentence
- Paper type: theory, empirical, survey, position paper?
- Research area and subfield

### 2. **Problem Identification** (The "Why")
- What specific problem does this paper address?
- Why is this problem important? What's at stake?
- What gaps in existing work does it fill?
- What's the research question or hypothesis?

### 3. **Solution/Approach** (The "How")
- Core contribution in 2-3 sentences
- Key technical innovations or insights
- Methodology: experimental, theoretical, mixed?
- Novel techniques or modifications to existing methods

### 4. **Evidence & Results** (The "Does it work?")
- Main experiments or theoretical proofs
- Key findings with concrete numbers
- Comparison to baselines (include actual metrics)
- Statistical significance and effect sizes

### 5. **Critical Analysis** (Strengths & Limitations)

**Strengths:**
- What does this paper do exceptionally well?
- Novel contributions to the field
- Methodological rigor
- Clarity of presentation

**Limitations:**
- Experimental: limited datasets, missing baselines, confounds
- Theoretical: assumptions, scope restrictions
- Generalizability concerns
- Missing ablations or analyses

### 6. **Research Context** (Connections)
- How does this fit into the broader literature?
- What prior work does it build on? (key citations)
- What future work does it enable?
- Related concurrent work?

### 7. **Practical Takeaways**
- If you were to apply this, what would you need to know?
- Reproducibility: is code/data available?
- Key implementation details or tricks
- When would you use this vs. alternatives?

---

## Output Structure

Present analysis in this format:

```markdown
# [Paper Title]

**Authors:** [Authors] | **Venue:** [Conference/Journal Year] | **Link:** [URL if available]

## 🎯 One-Sentence Summary
[Concise summary of contribution]

## 🔍 Problem & Motivation
[What problem? Why important? What's missing?]

## 💡 Key Contribution
[Core technical insight - the "aha!" moment]

## 🔬 Approach
[How they solve it - method overview]

## 📊 Results
[Key findings with numbers]

## ✅ Strengths
- [Strength 1]
- [Strength 2]

## ⚠️ Limitations
- [Limitation 1]
- [Limitation 2]

## 🔗 Research Context
[How it fits in the literature]

## 🛠️ Practical Notes
[Implementation details, reproducibility, when to use]

## 💭 Personal Notes
[Your thoughts, questions, ideas for follow-up]
```

---

## Reading Strategies

### For Different Paper Types:

**Theory Papers:**
- Focus on theorems, proofs, assumptions
- Check proof sketches vs. full proofs
- Identify which assumptions are restrictive

**Empirical Papers:**
- Datasets: size, domain, splits
- Experimental setup: hyperparameters, compute
- Ablations: what's necessary vs. auxiliary?

**Survey Papers:**
- Taxonomy: how do they organize the field?
- Coverage: what's included/excluded?
- Gaps: what's identified as future work?

**Position Papers:**
- Main argument and supporting evidence
- Counter-arguments considered?
- Implications if the position is correct

### Active Reading Prompts:

- "What would I need to reproduce this?"
- "What's the smallest change that would break this?"
- "How does this compare to [related work X]?"
- "What assumption is doing the most work here?"
- "If I had to explain this to someone in 2 minutes, what would I say?"

---

## Tools & Capabilities

When reading papers, leverage:

- **WebSearch**: Look up unfamiliar concepts, related papers, author backgrounds
- **WebFetch**: Pull in arXiv papers, citations, supplementary materials
- **Read**: Extract text from local PDFs (if available)
- **Write**: Save structured notes, summaries, questions
- **Code Analysis**: If paper includes code, analyze implementation details

---

## Multi-Paper Workflows

### Comparing Papers:
Create comparison tables:
| Aspect | Paper A | Paper B | Paper C |
|--------|---------|---------|---------|
| Problem | ... | ... | ... |
| Approach | ... | ... | ... |
| Results | ... | ... | ... |

### Literature Review:
1. Group papers by theme/approach
2. Identify evolution of ideas chronologically
3. Map citation relationships
4. Highlight open problems

### Replication Planning:
1. Extract all implementation details
2. List required datasets/resources
3. Identify ambiguities needing clarification
4. Create step-by-step reproduction plan

---

## Common Pitfalls to Avoid

❌ **Don't:**
- Uncritically accept claims - always look for evidence
- Ignore limitations section - it's often the most honest part
- Skip related work - context is crucial
- Accept vague language - demand concrete examples
- Forget to check if code/data are available

✅ **Do:**
- Read with specific questions in mind
- Take notes while reading (not just after)
- Check citation context (how do others use this paper?)
- Look for contradictions between intro and results
- Consider alternative explanations for results

---

## Example Usage

**User:** "Read this paper on attention mechanisms"

**Assistant:**
1. Fetches/reads paper
2. Follows 7-step analysis workflow
3. Outputs structured breakdown
4. Identifies key citations to explore
5. Suggests related papers or follow-up questions

**User:** "Compare these 3 papers on graph neural networks"

**Assistant:**
1. Reads all three papers
2. Creates comparison table
3. Identifies common themes and divergences
4. Highlights which paper to use when
5. Notes reproducibility status of each

---

## Tips for Effective Use

1. **Provide context:** Let me know your background and goals
   - "I'm trying to understand X for my project on Y"
   - "I'm preparing for a paper discussion in my reading group"

2. **Ask specific questions:** Guide the analysis
   - "Focus on the experimental setup"
   - "I want to understand the theoretical guarantees"

3. **Iterate:** Start with overview, then dive deeper
   - "Give me a quick summary first"
   - "Now let's dig into section 3.2 in detail"

4. **Connect to your work:** Make it actionable
   - "How would this apply to my problem with X?"
   - "What parts of this could I adapt for Y?"
