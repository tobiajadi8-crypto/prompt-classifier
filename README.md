# Prompt injection classifier

A small pretrained BERT encoder fine-tuned in PyTorch to flag potential prompt injections. The FastAPI service runs offline with the bundled 17.5 MB safetensors checkpoint. It returns a label, attack score, score for the selected label, and a flag indicating whether it truncated the input.

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

The input limit is 16,384 characters. Empty and whitespace-only requests return 422. The model scores the first 192 WordPiece tokens including special tokens. Inspect `truncated` when handling longer documents. The response scores are raw softmax outputs. The validation-selected attack threshold is 0.60.

For a container deployment, build with `docker build -t prompt-classifier .` and run `docker run --rm -p 127.0.0.1:8765:8000 prompt-classifier` after freeing the host port.

## Browser interface

To use the browser form with the running API, start `node frontend/server.mjs` in another terminal (Node.js 20+). Open [the local classifier](http://127.0.0.1:8781/), enter a prompt, and submit it for classification.

## Held-out experiment

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
