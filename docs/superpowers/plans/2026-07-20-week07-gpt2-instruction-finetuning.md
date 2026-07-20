# Week 07 GPT-2 Instruction Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and submit Choi Jinwoong's GPT-2 Medium instruction fine-tuning assignment with verified code, a real Colab GPU run, inference output, screenshots, and a pull request.

**Architecture:** Keep the provided from-scratch GPT-2 implementation and fill only the orchestration gaps in `train.py`, then use `inference.py` to reconstruct the same model and load the saved state dictionary. Local checks cover deterministic data preparation and tensor collation; Google Colab provides CUDA for the full two-epoch training and evidence capture.

**Tech Stack:** Python 3, PyTorch, tiktoken, TensorFlow checkpoint reader, matplotlib, Google Colab CUDA, Git, GitHub CLI/browser.

## Global Constraints

- Use `gpt2-medium (355M)` and the provided `instruction-data.json`.
- Train for exactly two epochs with the provided `train_model_simple` function.
- Keep source and screenshots under `week07/최진웅`.
- Do not commit `.pth` checkpoints or pretrained-weight cache directories.
- Use pull request title `[7주차 과제] 최진웅 과제 제출`.

---

### Task 1: Add Local Assignment Checks

**Files:**
- Create: `week07/최진웅/test_assignment.py`
- Test: `week07/최진웅/test_assignment.py`

**Interfaces:**
- Consumes: `InstructionDataset`, `custom_collate_fn`, and `format_input` from `train.py`.
- Produces: executable `unittest` checks for prompt formatting, shifted targets, ignored padding, and dataset size.

- [ ] **Step 1: Write the checks**

```python
import json
from pathlib import Path
import unittest

import torch

from train import custom_collate_fn, format_input


class AssignmentTests(unittest.TestCase):
    def test_dataset_has_expected_records(self):
        path = Path(__file__).with_name("instruction-data.json")
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        self.assertEqual(len(data), 1142)

    def test_format_input_includes_optional_input(self):
        text = format_input({"instruction": "Answer.", "input": "Context"})
        self.assertIn("### Instruction:\nAnswer.", text)
        self.assertIn("### Input:\nContext", text)

    def test_collate_shifts_and_ignores_extra_padding(self):
        inputs, targets = custom_collate_fn([[1, 2], [3]], pad_token_id=99)
        self.assertTrue(torch.equal(inputs, torch.tensor([[1, 2], [3, 99]])))
        self.assertTrue(torch.equal(targets, torch.tensor([[2, 99], [99, -100]])))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the checks before implementation**

Run: `python -m unittest test_assignment.py -v`

Expected: tests importing existing helpers pass; orchestration TODOs remain unimplemented and will be covered by source assertions added in Task 2.

- [ ] **Step 3: Commit the checks with the implementation in Task 2**

No standalone commit is made because the tests describe the same assignment deliverable as Task 2.

### Task 2: Implement Training Orchestration

**Files:**
- Modify: `week07/최진웅/train.py`
- Modify: `week07/최진웅/test_assignment.py`
- Test: `week07/최진웅/test_assignment.py`

**Interfaces:**
- Consumes: local `instruction-data.json`, `download_and_load_gpt2`, `GPTModel`, `load_weights_into_gpt`, and `train_model_simple`.
- Produces: dataloaders, a two-epoch fine-tuned Medium model, loss plot, and `gpt2-medium355M-sft-standalone.pth`.

- [ ] **Step 1: Add a failing source-completion check**

```python
    def test_train_source_has_no_assignment_todos(self):
        source = Path(__file__).with_name("train.py").read_text(encoding="utf-8")
        self.assertNotIn("# TODO:", source)
        self.assertIn("num_epochs=2", source)
        self.assertIn('CHOOSE_MODEL = "gpt2-medium (355M)"', source)
```

- [ ] **Step 2: Verify the source-completion check fails**

Run: `python -m unittest test_assignment.AssignmentTests.test_train_source_has_no_assignment_todos -v`

Expected: FAIL because `train.py` still contains `# TODO:`.

- [ ] **Step 3: Implement data preparation**

Load `instruction-data.json` relative to `__file__`, split at `int(len(data) * 0.85)` and `int(len(data) * 0.95)`, construct datasets, bind `custom_collate_fn` with `allowed_max_length=1024` and the selected device, then create train/validation/test dataloaders with batch size 8 and training shuffle enabled only for the training loader.

- [ ] **Step 4: Implement pretrained model loading**

Map the display name to checkpoint size `355M`, call `download_and_load_gpt2(model_size="355M", models_dir="gpt2")`, construct `GPTModel(BASE_CONFIG)`, load the checkpoint, set evaluation mode, and move the model to the selected device.

- [ ] **Step 5: Implement two-epoch fine-tuning and checkpoint saving**

Create `torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)`, set a deterministic seed, call `train_model_simple` with `num_epochs=2`, `eval_freq=5`, `eval_iter=5`, and an assignment-format start context, plot returned losses, and save the model state dictionary using the provided filename expression.

- [ ] **Step 6: Run local validation**

Run: `python -m unittest test_assignment.py -v`

Expected: all tests PASS.

Run: `python -m py_compile train.py inference.py gpt_download.py previous_chapters.py`

Expected: exit status 0 with no output.

- [ ] **Step 7: Commit source and tests**

```bash
git add week07/최진웅/train.py week07/최진웅/test_assignment.py
git commit -m "feat: complete GPT-2 instruction fine-tuning"
```

### Task 3: Configure World Cup Inference

**Files:**
- Modify: `week07/최진웅/inference.py`
- Modify: `week07/최진웅/test_assignment.py`
- Test: `week07/최진웅/test_assignment.py`

**Interfaces:**
- Consumes: `gpt2-medium355M-sft-standalone.pth` from Task 2.
- Produces: a generated answer to a South Korea Round-of-32 question.

- [ ] **Step 1: Add a failing inference prompt check**

```python
    def test_inference_asks_world_cup_question(self):
        source = Path(__file__).with_name("inference.py").read_text(encoding="utf-8")
        self.assertIn("2026 FIFA World Cup", source)
        self.assertIn("Round of 32", source)
```

- [ ] **Step 2: Verify it fails against the existing prompt**

Run: `python -m unittest test_assignment.AssignmentTests.test_inference_asks_world_cup_question -v`

Expected: FAIL because the existing prompt mentions criticism but not the Round of 32.

- [ ] **Step 3: Replace the inference entry**

Use the instruction `Why did South Korea fail to qualify for the Round of 32 in the 2026 FIFA World Cup?` with an empty input, retain greedy generation, and print both the question and cleaned response for readable evidence.

- [ ] **Step 4: Run local validation**

Run: `python -m unittest test_assignment.py -v && python -m py_compile inference.py`

Expected: all tests PASS and compilation exits 0.

- [ ] **Step 5: Commit inference**

```bash
git add week07/최진웅/inference.py week07/최진웅/test_assignment.py
git commit -m "feat: add World Cup model inference"
```

### Task 4: Run Colab Training and Capture Evidence

**Files:**
- Create: `week07/최진웅/training-result.png`
- Create: `week07/최진웅/inference-result.png`

**Interfaces:**
- Consumes: Task 2 and Task 3 source files and a Colab CUDA runtime.
- Produces: verified two-epoch checkpoint in the temporary runtime and submission screenshots in the repository.

- [ ] **Step 1: Open a signed-in Colab session and select GPU**

Open Colab, connect a T4-or-better GPU runtime, and verify with `torch.cuda.is_available()` and `nvidia-smi`.

- [ ] **Step 2: Transfer the working branch and install dependencies**

Make the branch files available to Colab and install `torch`, `tiktoken`, `tensorflow`, `matplotlib`, `requests`, and `tqdm` in versions compatible with the runtime.

- [ ] **Step 3: Run the local checks in Colab**

Run: `python -m unittest test_assignment.py -v`

Expected: all tests PASS.

- [ ] **Step 4: Train on CUDA**

Run: `python train.py`

Expected: device reports CUDA, pretrained 355M weights load, two epochs finish, and `gpt2-medium355M-sft-standalone.pth` is created. If CUDA memory is exhausted, change only the batch size from 8 to 4, rerun checks, and restart training.

- [ ] **Step 5: Run inference**

Run: `python inference.py`

Expected: output includes the World Cup question and a non-empty response about South Korea's qualification failure.

- [ ] **Step 6: Capture screenshots**

Capture the completed training output and the inference question/answer as PNG images, download them, and place them at the exact paths listed for this task. Verify both images are legible.

- [ ] **Step 7: Commit evidence**

```bash
git add week07/최진웅/training-result.png week07/최진웅/inference-result.png
git commit -m "docs: add GPT-2 training and inference results"
```

### Task 5: Final Verification and Pull Request

**Files:**
- Verify: `week07/최진웅/train.py`
- Verify: `week07/최진웅/inference.py`
- Verify: `week07/최진웅/test_assignment.py`
- Verify: `week07/최진웅/training-result.png`
- Verify: `week07/최진웅/inference-result.png`

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: pushed branch and assignment pull request.

- [ ] **Step 1: Verify repository contents**

Run: `git diff origin/main...HEAD --check`

Expected: exit status 0.

Run: `git status --short`

Expected: no output.

Run: `find week07/최진웅 -maxdepth 2 -type f -size +100M -print`

Expected: no output.

- [ ] **Step 2: Review the final diff**

Run: `git diff --stat origin/main...HEAD && git diff origin/main...HEAD -- week07/최진웅/train.py week07/최진웅/inference.py week07/최진웅/test_assignment.py`

Expected: only intended assignment implementation, checks, evidence, and planning documents appear.

- [ ] **Step 3: Push to a writable remote**

Authenticate GitHub, create or use Choi Jinwoong's fork if upstream push permission is unavailable, and push `codex/week07-choi-jinwoong`.

- [ ] **Step 4: Create the pull request**

Open a PR against `Likelion-MJU-AI-14th/likelion-mju-ai:main` with title `[7주차 과제] 최진웅 과제 제출` and a short body listing the TODO implementation, two-epoch Colab run, and attached inference screenshots.

- [ ] **Step 5: Verify the pull request**

Confirm the PR URL, base/head branches, exact title, changed files, and absence of checkpoints.
