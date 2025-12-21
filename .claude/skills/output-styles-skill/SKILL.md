---
name: output-styles-skill
description: Guide for creating and managing custom output styles in Claude Code. Use when user wants to modify Claude's behavior, create a new communication mode, or understand output styles.
---

# Output Styles Creation Guide

**When to use:** User asks about creating output styles, modifying Claude's behavior, or wants to customize response patterns for specific domains.

---

## What Are Output Styles?

**Output styles are Markdown files that replace Claude Code's default system prompt** to adapt Claude for different use cases beyond software engineering.

### The Mental Model:

Think of output styles as **personality modes** or **role presets**:
- **Default mode:** Software engineering assistant
- **Explain mode:** Teacher focused on deep understanding
- **Your custom mode:** Any role you define

They fundamentally change **how Claude communicates and what it prioritizes**.

---

## Output Style vs. Skill vs. CLAUDE.md

| Feature | Output Style | Skill | CLAUDE.md |
|---------|-------------|-------|-----------|
| **Purpose** | Change role/domain entirely | Add specific capabilities | Add project context |
| **Scope** | Replaces system prompt | Extends capabilities | Appends to context |
| **When Active** | Entire session | Auto-invoked when relevant | Always present |
| **Coding Mode** | Optional (via `keep-coding-instructions`) | Yes (coding-first) | Yes (coding-first) |
| **Best For** | Teacher, analyst, writer modes | PDF processing, commit help | Project conventions, APIs |

### Quick Decision Tree:

```
Do you want to change Claude's fundamental role?
├─ YES → Output Style
│   └─ "Be a teacher" / "Be a data analyst" / "Be a writer"
│
└─ NO → Is it coding-related?
    ├─ YES → Is it a specific capability?
    │   ├─ YES → Skill
    │   │   └─ "Review PRs" / "Generate commits" / "Write tests"
    │   └─ NO → CLAUDE.md
    │       └─ "Use our API conventions" / "Follow our style guide"
    │
    └─ NO → Output Style with keep-coding-instructions: false
        └─ "Write marketing copy" / "Analyze research papers"
```

---

## File Format

```markdown
---
name: style-name
description: Brief description shown in /output-style menu
keep-coding-instructions: false
---

# Your Custom System Prompt

You are [role/domain expert]...

## Core Behaviors

[Define how Claude should behave]

## Communication Style

[Specify tone, format, priorities]

## Specific Guidelines

[Domain-specific rules and patterns]
```

### Frontmatter Fields:

| Field | Required | Purpose | Example |
|-------|----------|---------|---------|
| `name` | No | Display name (defaults to filename) | `"Academic Researcher"` |
| `description` | No | Shown in menu | `"Analyze papers systematically"` |
| `keep-coding-instructions` | No | Keep default coding tools/behavior | `true` or `false` (default) |

---

## File Location

**Two options:**

1. **User-level** (global across all projects):
   ```
   ~/.claude/output-styles/my-style.md
   ```

2. **Project-level** (specific to current project):
   ```
   .claude/output-styles/my-style.md
   ```

Project-level styles take precedence over user-level if both exist.

---

## Creating Your First Output Style

### Step-by-Step Process:

#### 1. **Identify the Need**

Ask yourself:
- What domain am I working in? (research, writing, customer support)
- How should Claude communicate? (formal, casual, structured)
- What should Claude prioritize? (clarity, speed, depth)
- Do I still need coding tools? (set `keep-coding-instructions` accordingly)

#### 2. **Study Existing Examples**

Look at built-in styles for patterns. The "Explain" style is a good reference:

**Key components:**
- Core philosophy (the "why")
- Communication patterns (the "how")
- Specific rules (the "what")

#### 3. **Draft Your Style**

Start with a template:

```markdown
---
name: my-custom-style
description: [One sentence describing the purpose]
keep-coding-instructions: false
---

# [Role Name]

You are a [role] that helps users [primary goal].

## Core Philosophy

**[Main principle in bold]**

- [Key principle 1]
- [Key principle 2]
- [Key principle 3]

## Communication Patterns

**[Pattern category 1]:**
[Description and examples]

**[Pattern category 2]:**
[Description and examples]

## Style Guidelines

- [Specific rule 1]
- [Specific rule 2]
- [Specific rule 3]
```

#### 4. **Be Specific**

❌ **Vague:** "Be helpful and clear"
✅ **Specific:** "Start every response with a concrete example before explaining theory"

❌ **Vague:** "Write well"
✅ **Specific:** "Use active voice, short sentences (<20 words), and include 1-2 analogies per concept"

#### 5. **Test and Iterate**

1. Save your output style
2. Switch to it: `/output-style my-custom-style`
3. Test with real queries
4. Refine based on results
5. Switch back to default: `/output-style default`

---

## Common Output Style Patterns

### 1. **Teacher/Educator Mode**

**Goal:** Deep understanding over quick answers

```markdown
---
name: teacher
description: Educational mode focused on building understanding step-by-step
keep-coding-instructions: false
---

# Teaching Mode

You are an expert educator who builds deep understanding.

## Core Principles

- **Socratic method:** Ask questions to guide discovery
- **Concrete before abstract:** Always start with examples
- **Multiple representations:** Explain through different lenses

## Communication Pattern

1. Start with a motivating question or problem
2. Present a concrete example with actual numbers
3. Extract the general principle
4. Provide an analogy from everyday life
5. Highlight common misconceptions

## Style

- Use simple language (5th-grade reading level)
- Break complex ideas into digestible chunks
- Include practice problems or thought exercises
- Celebrate "aha!" moments
```

### 2. **Research Analyst Mode**

**Goal:** Critical evaluation and evidence-based insights

```markdown
---
name: research-analyst
description: Systematic research analysis with critical evaluation
keep-coding-instructions: false
---

# Research Analyst

You are a methodical researcher who evaluates evidence critically.

## Core Behaviors

- **Evidence-first:** Every claim needs supporting data
- **Skeptical by default:** Question assumptions and methods
- **Comprehensive:** Consider multiple perspectives

## Analysis Framework

1. **Context:** Background and motivation
2. **Methods:** How was this studied?
3. **Results:** What was found? (with numbers)
4. **Limitations:** What are the caveats?
5. **Implications:** What does this mean?

## Output Format

Use structured markdown with:
- Clear section headers
- Bullet points for findings
- Tables for comparisons
- Citations in [Author, Year] format
```

### 3. **Content Writer Mode**

**Goal:** Engaging, audience-appropriate content

```markdown
---
name: content-writer
description: Create engaging content tailored to audience and platform
keep-coding-instructions: false
---

# Content Writer

You craft compelling content optimized for the target audience.

## Writing Principles

- **Audience-first:** Always ask "Who is reading this?"
- **Hook early:** First sentence must grab attention
- **Show, don't tell:** Use stories and examples

## Content Structure

1. **Hook:** Attention-grabbing opening
2. **Promise:** What will the reader gain?
3. **Deliver:** Main content with examples
4. **Call-to-action:** What should they do next?

## Style Guidelines

- Active voice (>90% of sentences)
- Vary sentence length: short (impact), long (flow)
- Use power words: discover, proven, secret, essential
- Include subheadings every 2-3 paragraphs
- End with a memorable takeaway
```

### 4. **Hybrid: Teacher + Coding**

**Goal:** Teach programming concepts while keeping tools

```markdown
---
name: coding-teacher
description: Teach programming concepts with hands-on examples
keep-coding-instructions: true  # ← Key difference!
---

# Coding Educator

You teach programming through concrete examples and hands-on practice.

## Teaching Approach

- **Write code first, explain after:** Show working examples immediately
- **Iterative complexity:** Start simple, add features incrementally
- **Bugs as learning:** Use errors as teaching moments

## Code Examples

- Include line-by-line explanations
- Show "before/after" comparisons
- Highlight common pitfalls with warnings
- Provide exercises for practice

## Communication Style

- Use analogies from everyday life
- Draw ASCII diagrams for data structures
- Compare to concepts they already know
- Celebrate working code with enthusiasm
```

---

## Advanced Techniques

### 1. **Context-Aware Instructions**

Make your style adapt to different scenarios:

```markdown
## Adaptive Behavior

When user provides:
- **Code:** Analyze patterns, suggest improvements, explain design choices
- **Questions:** Break down into subquestions, answer systematically
- **Data:** Visualize patterns, compute statistics, identify anomalies
- **Papers:** Extract key claims, evaluate evidence, summarize findings
```

### 2. **Structured Output Formats**

Define templates for consistency:

```markdown
## Response Template

Every analysis should follow:

```
# [Title]

## 📋 Summary
[2-3 sentence overview]

## 🔍 Deep Dive
[Detailed analysis]

## 💡 Key Takeaways
- [Takeaway 1]
- [Takeaway 2]

## ❓ Questions to Consider
- [Question 1]
- [Question 2]
```
```

### 3. **Persona-Based Styles**

Create distinct personalities:

```markdown
# Skeptical Scientist

You are a rigorous scientist who demands evidence and clear reasoning.

## Personality Traits

- **Skeptical:** Challenge unsupported claims
- **Precise:** Use exact language, avoid vagueness
- **Quantitative:** Prefer numbers over adjectives
- **Methodical:** Follow systematic analysis steps

## Language Patterns

- "What's the evidence for...?"
- "How do we know that...?"
- "Let's quantify..."
- "The data shows that..."
```

---

## Testing Your Output Style

### Quality Checklist:

✅ **Clarity:**
- [ ] Can a new user understand what this style does?
- [ ] Are instructions specific rather than vague?
- [ ] Are there concrete examples?

✅ **Completeness:**
- [ ] Does it cover communication tone?
- [ ] Does it specify output format?
- [ ] Does it handle edge cases?

✅ **Consistency:**
- [ ] Will Claude behave predictably?
- [ ] Are there contradictory instructions?

✅ **Value:**
- [ ] Does it solve a real need?
- [ ] Is it better than the default for this use case?

### Test Scenarios:

Try your style with:
1. A simple query (does it handle basics?)
2. A complex query (does it maintain quality?)
3. An ambiguous query (does it ask good questions?)
4. An edge case (does it fail gracefully?)

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Too Vague

**Bad:**
```markdown
You are helpful and provide good answers.
```

**Good:**
```markdown
You provide structured answers with:
1. A concrete example
2. General principle
3. Common pitfalls
4. When to use vs. not use
```

### ❌ Mistake 2: Contradictory Instructions

**Bad:**
```markdown
- Be concise and brief
- Provide comprehensive detailed explanations
```

**Good:**
```markdown
- Start with a 1-sentence summary
- Then provide detailed breakdown for those who want depth
```

### ❌ Mistake 3: Overloading One Style

If your style tries to do too many things, split it:
- `research-paper-reader.md`
- `research-paper-writer.md`
- `research-grant-writer.md`

Each should have a focused purpose.

### ❌ Mistake 4: Forgetting `keep-coding-instructions`

If you need file operations, code execution, or git commands, set:
```yaml
keep-coding-instructions: true
```

Without this, Claude loses access to coding tools.

---

## Real-World Examples

### Example 1: Customer Support Style

```markdown
---
name: customer-support
description: Empathetic customer support with clear solutions
keep-coding-instructions: false
---

# Customer Support Specialist

You help customers solve problems with empathy and clarity.

## Response Framework

1. **Acknowledge:** Show you understand their issue
2. **Clarify:** Ask questions if anything is unclear
3. **Solve:** Provide step-by-step solution
4. **Verify:** Confirm it works for them

## Tone Guidelines

- **Empathetic:** "I understand how frustrating that must be"
- **Clear:** Use simple language, avoid jargon
- **Positive:** Focus on solutions, not problems
- **Patient:** Never rush or dismiss concerns

## Format

Use this structure:

**Issue:** [Restate their problem]
**Solution:** [Step-by-step fix]
**Why this works:** [Brief explanation]
**If this doesn't help:** [Next steps]
```

### Example 2: Data Analyst Style

```markdown
---
name: data-analyst
description: Data-driven insights with statistical rigor
keep-coding-instructions: true  # Need code for analysis
---

# Data Analyst

You analyze data systematically and communicate insights clearly.

## Analysis Workflow

1. **Understand the question:** What are we trying to learn?
2. **Examine the data:** Sample size, distributions, outliers
3. **Choose methods:** Statistical tests, visualizations
4. **Run analysis:** Code + results
5. **Interpret:** What does it mean? What's the confidence?

## Communication Rules

- Lead with the insight, then show the evidence
- Always include uncertainty (p-values, confidence intervals)
- Use visualizations for patterns
- Highlight limitations and assumptions

## Code Style

- Comment every analysis step
- Print intermediate results
- Create clear visualizations
- Save results to files for reference
```

---

## Tips for Success

### 1. **Start Small**

Begin with a simple style for one specific use case. Expand later.

### 2. **Copy Good Patterns**

Study effective writing styles you admire:
- Technical documentation (Stripe, Cloudflare)
- Educational content (3Blue1Brown, Khan Academy)
- Professional communication (your favorite authors)

### 3. **Version Control**

Keep your styles in git to track changes:
```bash
git add .claude/output-styles/
git commit -m "Add research analyst output style"
```

### 4. **Share with Team**

Project-level styles can be shared via git:
```bash
# Team member pulls latest
git pull
# Now they have your output styles
```

### 5. **Iterate Based on Use**

Track what works:
- Which instructions does Claude follow well?
- Which parts get ignored?
- What produces the best results?

Refine over time.

---

## Usage Commands

### Switching Styles:

```bash
# View available styles
/output-style

# Switch to a specific style
/output-style explain

# Return to default (software engineering)
/output-style default
```

### File Management:

```bash
# Create new style
vim ~/.claude/output-styles/my-style.md

# List all styles
ls ~/.claude/output-styles/
ls .claude/output-styles/

# Edit existing
vim .claude/output-styles/explain.md
```

---

## Troubleshooting

### Problem: Claude Doesn't Follow the Style

**Possible causes:**
1. Instructions too vague → Be more specific
2. Contradictory instructions → Review for conflicts
3. Style file not loaded → Check file path and syntax

**Solution:** Add concrete examples of desired behavior

### Problem: Style Works Sometimes, Not Always

**Cause:** User messages might override style instructions

**Solution:** Make critical instructions more emphatic:
```markdown
## CRITICAL RULES

You MUST:
- [Non-negotiable rule 1]
- [Non-negotiable rule 2]

NEVER:
- [What to avoid]
```

### Problem: Lost Coding Capabilities

**Cause:** `keep-coding-instructions: false` removes tools

**Solution:** Set `keep-coding-instructions: true` or use Bash/Read/Write tools explicitly in your style

---

## Example: Creating a Paper Reading Style

**Wait! Paper reading should be a SKILL, not an output style.**

**Why?**
- It's a specific task, not a fundamental mode
- You want to invoke it when needed, not change your entire session
- You'll want to switch contexts (reading → coding → searching)

**Output styles** = Change who Claude is
**Skills** = Add what Claude can do

For paper reading, create `.claude/skills/paper-reading/SKILL.md` instead!

---

## Next Steps

1. **Identify a use case** where default Claude isn't optimal
2. **Draft your style** using the templates above
3. **Test thoroughly** with real scenarios
4. **Iterate** based on results
5. **Share** if it's useful for others

Remember: Output styles are powerful but not always the answer. Consider skills and CLAUDE.md for many use cases!
