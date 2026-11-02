# Pull request

## What this changes

<!-- One or two sentences. What is different after this lands? -->

## Why

<!-- The problem this solves, and the issue it closes. -->

Closes #

## How it was verified

<!--
Say how you tested it. Numbers beat adjectives: tok/s, time to live endpoint,
peak memory, and the accelerator they came from.

If you ran a Kaggle session, link the run. If you ran a local harness, paste
the command and its output.
-->

- [ ] Tested on the accelerator this touches (TPU v5e-8 / T4 x2 / GPU / Modal / GCP TPU VM)
- [ ] Numbers in the README are from this change, not from an earlier one
- [ ] Notebooks regenerated from their kernel scripts (`tools/pack_notebook.py`), if a kernel changed
- [ ] No credentials, API keys or dataset slugs that are not already public in this repository

## Notes for the reviewer

<!--
Anything you want a second opinion on: a risky path, a measurement you are
unsure of, a part you would rather not review.
-->
