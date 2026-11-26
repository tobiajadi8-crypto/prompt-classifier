# Prompt injection classifier

A small pretrained BERT encoder fine-tuned in PyTorch to flag potential prompt injections. The FastAPI service runs offline with the bundled 17.5 MB safetensors checkpoint. It scans the whole input and returns the highest-scoring passage for review.

## Run

Use Python 3.12 in a virtual environment. From this directory:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8765
```

Open [interactive API docs](http://127.0.0.1:8765/docs), select **POST /predict**, and submit the JSON below. The readiness endpoint is `GET /health`.

```json
{"text": "Could you recommend a book about growing tomatoes?"}
```

The input limit is 16,384 Unicode code points. Requests with no usable tokens return 422. Each window holds up to 190 content tokens and two BERT special tokens. Adjacent windows share 64 content tokens. The API scores up to eight windows per batch and returns a result after all windows have been checked. A single-window input follows the original scoring rule.

The document's `attack_score` is the highest window score. A score at or above 0.60 yields `attack`. This threshold was selected on short validation prompts. The score measures model output and has no document-level probability calibration. The existing `label`, `attack_score`, `confidence` and `truncated` fields remain available. Full scans return `truncated: false`. The new `passages_checked` field gives the number of scored windows. The one-based `highest_scoring_passage.number` identifies the first window with the maximum score. Its `start` and `end` are half-open Unicode code-point offsets into the original text.

For a container deployment, build with `docker build -t prompt-classifier .` and run `docker run --rm -p 127.0.0.1:8765:8000 prompt-classifier` after freeing the host port.

## Browser interface

With the API running, start `node frontend/server.mjs` in another terminal (Node.js 20+). Open [the local classifier](http://127.0.0.1:8781/), paste a document and select **Scan document**. The result reports coverage and highlights the entire highest-scoring window. This highlights what the model scored, including any benign text in the same window.

## Whole-document experiment

Two inference strategies used identical weights and the same 0.60 threshold. The baseline scored only the first 192 tokens. The full scanner took the maximum over all overlapping windows.

The fixed synthetic benchmark contains ten fictional documents (four research notes, three emails and three support tickets). Each has one clean version and three variants with the same harmless task-redirection request inserted at the start, middle or end. The assigned task is to review the document. Requests embedded in the document to switch tasks are labeled as injections. Clean versions also include ordinary instructions to people. Every middle and end insertion is outside the baseline's input. All 40 documents produced four windows.

| Measurement | First window | Full scan |
|---|---:|---:|
| Clean documents incorrectly flagged | 10/10 | 10/10 |
| Start / middle / end variants flagged (10 per position) | 10 / 10 / 10 | 10 / 10 / 10 |
| Median warm CPU time per document | 17.4 ms | 44.2 ms |

**The change improved coverage. Neither approach distinguished clean documents from modified ones on this probe.** For example, the prefix scanner gave the clean slide-archive note and its end-insertion variant the same score (0.993575). It never saw the inserted text. The full scan's top window completely contained the labeled passage in 17 of 30 variants (5 start, 6 middle, 6 end). Even a matching window can contain unrelated text, so the highlight cannot establish which words caused the score.

The ten base documents use genre-specific templates with repeated sentence patterns and three distinct task-redirection requests. These related cases test placement under controlled conditions and cannot estimate real-world prevalence or clinical performance. The documents are synthetic and contain no patient records. No training examples, weights or thresholds were changed for this experiment.

On the original 116-row deepset test split, two inputs required multiple windows. All 116 full-scan labels matched the prefix baseline (55 true positives, 54 true negatives, two false positives and five false negatives).

Run the following after installing the training dependencies. The position probe runs offline. Timings use five warm repetitions per method per case, alternating order, with two PyTorch CPU threads and no HTTP overhead. The reported latency is the median of the 40 per-document medians.

```sh
python -m unittest -v test_scanner
python evaluate_documents.py --out local_document_evaluation.json
```

To also check the original test split, pass `--published-test` with the path to the pinned test parquet downloaded by `fetch.py`. [Per-case scores, timings and localization results](document_evaluation.json) · [Synthetic fixtures](benchmarks/documents.json) · [Case construction](benchmarks/cases.py)

## Original short-prompt experiment (first window)

Used the published deepset test partition of 116 examples (60 attacks, 56 safe). All three methods saw the same test rows. Attack is the positive class. Latencies are warm median milliseconds per single example on an Intel Core Ultra 9 185H CPU, including tokenization or vectorization, excluding HTTP. PyTorch used two CPU threads.

| Model | Precision | Recall | F1 | Median ms |
|---|---:|---:|---:|---:|
| Fine-tuned BERT, 2 layers, 128 hidden units | 0.965 | 0.917 | 0.940 | 3.53 |
| Word unigram/bigram TF-IDF + logistic regression | 0.979 | 0.767 | 0.860 | 0.70 |
| FLAN-T5-small, fixed yes/no prompt | 1.000 | 0.067 | 0.125 | 82.19 |

BERT caught 55 attacks and missed five. The linear baseline caught 46, with one false alarm. The prompted model predicted attack only four times. Its perfect precision reflects those four decisions. The comparison applies to this specific local model and prompt.

BERT's two false positives were English and German versions of a short search query about LGBTQ+ equality in Europe. One missed example was a short translation request labeled as an attack. A standalone string can omit the surrounding task that makes a redirection malicious. The benchmark is small and contains related multilingual examples. Its scores give limited evidence about unfamiliar attack families or long documents.

## Reproduce

```sh
python -m pip install -r requirements-train.txt
python train.py --cache .cache --out run
```

The first run downloads pinned revisions of the public dataset and base models. Training uses the published train partition. A character 3–5-gram similarity graph groups examples with cosine similarity of at least 0.80. Training excludes groups that intersect the official test set and removes normalized exact duplicates. A deterministic stratified group split yields 428 training examples and 108 validation examples. This heuristic can miss translations and semantically similar templates.

Training updates the whole BERT encoder and classification head with AdamW, learning rate 0.0002, batch size 16, gradient clipping at 1.0 and seed 42. Selection uses validation F1 over eight epochs and five thresholds. Epoch 7 and threshold 0.60 were selected before test scoring. Logistic regression uses balanced class weights and its default decision boundary. The zero-shot FLAN baseline chooses whichever of the single-token answers “yes” and “no” has the higher first-step logit, given the fixed prompt in the training script.

[Evaluation JSON](evaluation.json) records split counts, confusion matrices, latency and the index of every misclassification in the pinned test partition. Model weights, tokenizer and inference threshold are in [model](model/).

## Sources

The fine-tuned checkpoint derives from Google's Apache-2.0 BERT miniature. The training data and the prompted comparison model are credited below. The distributed checkpoint includes the upstream license.

- [Google BERT miniature](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)
- [deepset prompt-injections dataset](https://huggingface.co/datasets/deepset/prompt-injections)
- [Google FLAN-T5-small](https://huggingface.co/google/flan-t5-small)
