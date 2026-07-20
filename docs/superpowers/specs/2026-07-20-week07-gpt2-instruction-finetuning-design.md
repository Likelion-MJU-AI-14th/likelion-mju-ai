# Week 07 GPT-2 Instruction Fine-Tuning Design

## Goal

Complete Choi Jinwoong's Week 07 assignment by implementing the provided GPT-2 Medium instruction fine-tuning template, running two training epochs on Google Colab GPU, asking a 2026 World Cup question, and submitting code and screenshots through a pull request.

## Scope

- Modify only the assignment implementation under `week07/최진웅` plus this planning documentation.
- Implement every TODO in `train.py` without replacing the provided model implementation.
- Use the provided `instruction-data.json` as the sole fine-tuning dataset.
- Run `gpt2-medium (355M)` for exactly two epochs.
- Do not commit model checkpoints or downloaded pretrained-weight caches.
- Add screenshots that demonstrate the implemented code and the model's generated answer.

## Training Design

Load all 1,142 JSON records and preserve their existing order. Use the first 85% for training, the next 10% for validation, and the remainder for testing, matching the chapter's reference workflow. Construct `InstructionDataset` instances with the GPT-2 tokenizer and use the provided `custom_collate_fn` with padding token `50256`, ignore index `-100`, context limit `1024`, and the selected runtime device. Use shuffled training batches and deterministic validation/test batches.

Build `GPTModel` with the supplied Medium configuration, download OpenAI's 355M checkpoint through `download_and_load_gpt2`, load it with `load_weights_into_gpt`, switch to evaluation mode, and move it to the selected CUDA device. Fine-tune all parameters with AdamW using learning rate `5e-5` and weight decay `0.1` for two epochs through `train_model_simple`. Save the final state dictionary as `gpt2-medium355M-sft-standalone.pth` in the Colab runtime only.

## Inference Design

Load the saved state dictionary into the same Medium architecture, switch to evaluation mode, and generate a response with the provided greedy generation helper. The prompt will ask why South Korea failed to reach the 2026 World Cup Round of 32, which is directly represented by consistent examples near the beginning of the supplied dataset. Print the cleaned response to the notebook output.

## Colab Workflow and Evidence

Upload or clone the assignment files into a GPU-enabled Colab session, install compatible Python dependencies, and execute training from `week07/최진웅`. If batch size 8 exhausts GPU memory, reduce only the batch size while keeping the model, dataset, context limit, optimizer, and two-epoch requirement unchanged. Capture readable screenshots of the completed TODO sections and the final inference prompt/output. Store the screenshots under `week07/최진웅` with descriptive PNG filenames.

## Validation and Error Handling

Before full training, compile both Python entry points, validate the dataset split sizes and coverage, and exercise `custom_collate_fn` on a small synthetic batch. During Colab execution, verify CUDA availability, successful pretrained-weight loading, two completed epochs, checkpoint creation, and non-empty inference output. Dependency, download, CUDA-memory, or authentication failures must be surfaced rather than represented as successful training.

## Git Submission

Work on branch `codex/week07-choi-jinwoong`. Commit source and screenshots but exclude the generated checkpoint and pretrained-weight cache. Push to a user-writable remote or fork, then open a pull request targeting `Likelion-MJU-AI-14th/likelion-mju-ai` with the exact title `[7주차 과제] 최진웅 과제 제출`.
