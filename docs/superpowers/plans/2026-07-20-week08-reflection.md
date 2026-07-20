# Week 08 Reflection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Choi Jinwoong's Week 08 Markdown template with the supplied reflection and submit it through a dedicated pull request.

**Architecture:** This is a single-document change. Preserve repository metadata and headings, replace placeholders with user-authored prose, then verify content and Git scope.

**Tech Stack:** Markdown, Git, GitHub.

## Global Constraints

- Modify final diff only at `week08/최진웅/week08_최진웅.md`.
- Preserve the author name and assigned video URL.
- Use PR title `[8주차 과제] 최진웅 과제 제출`.

---

### Task 1: Replace the Reflection Template

**Files:**
- Modify: `week08/최진웅/week08_최진웅.md`

**Interfaces:**
- Consumes: the user's three supplied reflection sections.
- Produces: a complete Markdown assignment with no placeholder prose.

- [ ] **Step 1: Verify the template currently contains placeholders**

Run: `rg -n '\.\.\.' week08/최진웅/week08_최진웅.md`

Expected: four placeholder lines are found.

- [ ] **Step 2: Replace sections 1 through 3 and remove section 4**

Use the user's exact prose under the matching headings while preserving the title, name, and video URL.

- [ ] **Step 3: Validate content and scope**

Run: `test "$(rg -c '^## [123]\.' week08/최진웅/week08_최진웅.md)" = "3" && ! rg -n '\.\.\.' week08/최진웅/week08_최진웅.md`

Expected: exit status 0.

Run: `test "$(git diff --name-only origin/main...HEAD | wc -l | tr -d ' ')" = "1"`

Expected: exit status 0 after internal planning files are removed.

- [ ] **Step 4: Commit, push, and create PR**

Commit the reflection, push `codex/week08-choi-jinwoong` to the fork, and open a PR to organization `main` with the required title.
