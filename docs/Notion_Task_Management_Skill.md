# Notion Task Management Analysis Skill

## Purpose
This skill enables efficient analysis and management of Junjie's research tasks stored in Notion.

---

## 🔗 Key Notion Resources

### Primary Task Database
- **URL**: https://www.notion.so/junjiezhou1122/2a91655b750281b08edff8478e09673a
- **View with filters**: https://www.notion.so/junjiezhou1122/2a91655b750281b08edff8478e09673a?v=2a91655b7502817fb7a9000c2baf8a3f
- **Type**: Database containing all tasks
- **Parent**: Projects & Tasks page (https://www.notion.so/2a91655b7502800da99fe3ec391a886c)

### Projects Database
- **URL**: https://www.notion.so/2a91655b750281efbe47fea6ea21c5d6
- **Type**: Database containing all projects
- **Collection ID**: `collection://2a91655b-7502-81cf-9201-000bd586b014`

### Tasks Data Source
- **Collection ID**: `collection://2a91655b-7502-812b-ad01-000ba382c060`
- This is the underlying data source for all tasks

---

## 📊 Task Database Schema

### Properties
- **Task name** (title): The task title
- **Status** (status): Not Started | In Progress | Done | Archived
- **Project** (relation): Links to Projects database
- **Priority** (select): Low | Medium | High
- **Assignee** (person): Person assigned
- **Due** (date): Due date
- **Completed on** (date): Completion date
- **Tags** (multi_select): Mobile, Website, Improvement, Research, etc.
- **Parent-task** (relation): For subtasks
- **Sub-tasks** (relation): Child tasks
- **Delay** (formula): Calculated delay metric

---

## 🔍 How to Query Tasks

### Method 1: Search All Tasks
```
notion___notion-search with:
- query: "task" or relevant keyword
- query_type: "internal"
- data_source_url: "collection://2a91655b-7502-812b-ad01-000ba382c060"
```

### Method 2: Fetch Specific Task by ID
```
notion___notion-fetch with:
- id: <task_page_id>
```

### Method 3: Search by Project
```
notion___notion-search with:
- query: <project_name>
- query_type: "internal"
- data_source_url: "collection://2a91655b-7502-81cf-9201-000bd586b014"
```

---

## 🏗️ Project Structure

### Active Projects (as of Dec 2025)

1. **Reg2Rg_Improve** 
   - Status: In Progress
   - Focus: Model improvement
   - Key tasks: LIT training

2. **Medical_image_Evaluation**
   - Status: In Progress
   - Focus: Intervention-based evaluation paradigm
   - Key tasks: Core problem, Noisy Image (active), what to do, rrg model

3. **R4S** (Research for Software)
   - Status: In Progress
   - Focus: Agentic coding, not reinventing wheels
   - Multiple setup and development tasks

4. **Other Projects**: Speech, Programming, Agents, Evaluation, Collective intelligence, Learning theory, Safety

---

## 📝 Analysis Workflow

### Step 1: Get Overview
1. Fetch Projects database to see all projects and their status
2. Search Tasks data source to get task count and distribution

### Step 2: Detailed Analysis
1. Fetch individual project pages to see descriptions and context
2. For each project, search related tasks using project relation
3. Fetch key tasks to see detailed content

### Step 3: Identify Priorities
- Look for tasks with Status = "In Progress"
- Check Priority field (High > Medium > Low)
- Review Due dates for upcoming deadlines
- Identify tasks with extensive content (indicates active thinking)

### Step 4: Summarize
- Group by Project
- Separate by Status (In Progress, Not Started, Done)
- Highlight tasks with substantial content/notes
- Note any blockers or dependencies

---

## 🎯 Key Task Patterns

### Tasks with Detailed Content
These indicate active research thinking and should be prioritized:
- **Core problem**: Deep evaluation paradigm research
- **what to do**: Strategic planning with 3 research directions
- **Noisy Image**: Active experimental design
- **Ideas**: Extensive architecture brainstorming
- **LIT**: Training configuration ready

### Empty/Placeholder Tasks
These need definition or can be cleaned up:
- Generic "Task" entries with no content
- Tasks created but not yet scoped

---

## 💡 Quick Commands for Future Sessions

### Get All Projects
```
notion___notion-fetch with id: "2a91655b750281efbe47fea6ea21c5d6"
```

### Get All Tasks Overview
```
notion___notion-fetch with id: "2a91655b750281b08edff8478e09673a"
```

### Search In-Progress Tasks
```
notion___notion-search with:
- query: "In Progress" or "progress"
- query_type: "internal"
- data_source_url: "collection://2a91655b-7502-812b-ad01-000ba382c060"
```

### Search by Research Topic
Common topics to search:
- "evaluation" - for evaluation framework tasks
- "LIT" - for Linear Interpolation Transformer work
- "R4S" - for Research for Software tasks
- "noisy" or "noise" - for image robustness work
- "training" - for training-related tasks

---

## 🔄 Best Practices

1. **Always use parallel calls** when fetching multiple independent items
2. **Start with Projects first**, then drill into tasks
3. **Look for recent timestamps** to identify active work
4. **Check task content length** - substantial content = active thinking
5. **Note the hierarchy**: Project → Tasks → Subtasks
6. **Preserve exact URLs** for easy reference

---

## 🚀 Common Analysis Requests

### "Summarize all my tasks"
1. Fetch Projects database
2. Search tasks by project relation
3. Group by Status and Priority
4. Highlight tasks with content

### "What should I work on next?"
1. Filter Status = "In Progress"
2. Check Priority = "High"
3. Look for tasks with Due dates
4. Review detailed content for readiness

### "Show progress on [Project]"
1. Fetch specific project page
2. Search tasks with that project relation
3. Count tasks by status
4. List key accomplishments (Done tasks)

---

## 📌 Notes

- User primarily works on medical imaging AI research
- Strong focus on evaluation paradigms and model understanding
- Current active work: LIT probe and noise-based evaluation
- Prefers detailed documentation in task content
- Uses Notion as research notebook + task manager hybrid

---

## Last Updated
2025-12-19
