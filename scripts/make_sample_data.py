"""Generate a synthetic research corpus and a labelled evaluation set.

Why synthetic: the retrieval code, the metrics and the factory all need a corpus
with *known* ground truth that is small enough to ship in a repository and free
of licensing questions. Every document here is an original short summary of a
well-known paper, written so that specific sentences are uniquely identifyable
by a question - which is exactly what a relevance label needs to be.

The evaluation labels are **sentence anchors**, not chunk ids. The script writes
the full sentence it expects to find, and ``scripts/evaluate.py`` resolves it to
whichever chunk contains it. That keeps the label valid when chunk size changes;
it would otherwise be silently invalidated by any tuning of ``chunk_words``.

Usage::

    python scripts/make_sample_data.py            # corpus + testset
    python scripts/make_sample_data.py --pdf      # also emit real PDFs (needs reportlab)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CORPUS_DIR = REPO_ROOT / "data" / "sample"
PDF_DIR = REPO_ROOT / "data" / "sample_pdf"
TESTSET = REPO_ROOT / "eval" / "testset.jsonl"

# --------------------------------------------------------------------------
# The corpus. {placeholder} values are filled per document at write time.
# --------------------------------------------------------------------------
DOCS: list[dict] = [
    {
        "key": "attention",
        "title": "Attention Is All You Need",
        "year": 2017,
        "venue": "NeurIPS",
        "authors": "Vaswani et al.",
        "category": "natural language processing",
        "abstract": (
            "We propose the Transformer, a sequence transduction architecture based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "The model reaches state of the art translation quality while being substantially "
            "more parallelisable than recurrent networks."
        ),
        "sections": [
            (
                "Architecture",
                [
                    "The Transformer stacks an encoder and a decoder, each built from identical layers "
                    "of multi-head self-attention followed by a position-wise feed-forward network.",
                    "Because recurrence is removed, the model has no inherent notion of token order, "
                    "so sinusoidal positional encodings are added to the input embeddings.",
                    "Multi-head attention projects queries, keys and values into several subspaces so "
                    "that different heads can attend to different kinds of relationships.",
                ],
            ),
            (
                "Training",
                [
                    "The model is trained on the WMT 2014 English to German and English to French "
                    "translation benchmarks using the Adam optimiser with a warmup schedule.",
                    "Dropout and label smoothing are used as regularisation, and attention weights are "
                    "visualised to inspect what individual heads learn.",
                ],
            ),
            (
                "Results",
                [
                    "On WMT 2014 English to German the big Transformer reaches 28.4 BLEU, more than two "
                    "BLEU above the best previously reported ensembles.",
                    "Training the big configuration on eight GPUs takes about three and a half days, "
                    "a small fraction of the cost of comparable recurrent systems.",
                ],
            ),
            (
                "Complexity",
                [
                    "Self-attention costs quadratic time in the sequence length, which is why long context "
                    "variants use sparse or sliding window attention.",
                    "Every additional context token increases the cost of attention, so the quadratic term "
                    "is the usual motivation for efficient attention research.",
                ],
            ),
            (
                "Follow-up work",
                [
                    "Decoder-only variants keep the same attention mechanism but drop the encoder and train "
                    "on a next token prediction objective.",
                    "These models scale predictably with parameters and data, which is what makes the "
                    "architecture the default backbone for language modelling.",
                ],
            ),
        ],
    },
    {
        "key": "bert",
        "title": "Pretraining of Deep Bidirectional Transformers",
        "year": 2019,
        "venue": "NAACL",
        "authors": "Devlin et al.",
        "category": "natural language processing",
        "abstract": (
            "We introduce a bidirectional encoder pretrained with a masked language modelling "
            "objective. The pretrained model can be fine-tuned with one additional output layer to "
            "reach strong results across a wide range of sentence-level and token-level tasks."
        ),
        "sections": [
            (
                "Pretraining objective",
                [
                    "The masked language model objective randomly masks fifteen percent of input tokens "
                    "and trains the encoder to recover them from their two-sided context.",
                    "A next sentence prediction objective is trained jointly, although later work shows "
                    "it contributes little to downstream accuracy.",
                ],
            ),
            (
                "Fine-tuning",
                [
                    "Fine-tuning is inexpensive because the same pretrained parameters are reused for "
                    "every downstream task with only small task-specific heads added.",
                    "The model sets new state of the art results on the GLUE benchmark and on the "
                    "SQuAD question answering dataset.",
                ],
            ),
            (
                "Limitations",
                [
                    "Masked pretraining is compute hungry because the encoder is large and each step only "
                    "supervises fifteen percent of the input tokens.",
                    "Later encoders drop the next sentence prediction objective, train longer, and use "
                    "more data, which improves downstream accuracy at the same architecture size.",
                ],
            ),
        ],
    },
    {
        "key": "ddpm",
        "title": "Denoising Diffusion Probabilistic Models",
        "year": 2020,
        "venue": "NeurIPS",
        "authors": "Ho et al.",
        "category": "generative vision",
        "abstract": (
            "We present high quality image synthesis using diffusion probabilistic models, a class "
            "of latent variable models inspired by non-equilibrium thermodynamics. Training amounts "
            "to learning to reverse a gradual noising process."
        ),
        "sections": [
            (
                "Method",
                [
                    "A forward process gradually adds Gaussian noise to an image over a fixed number of "
                    "timesteps until the signal is destroyed.",
                    "The reverse process is parameterised by a neural network trained to predict the "
                    "noise that was added at each step, which makes the training objective a simple "
                    "weighted mean squared error.",
                    "Sampling starts from pure Gaussian noise and repeatedly applies the learned "
                    "denoising step to produce an image.",
                ],
            ),
            (
                "Results",
                [
                    "The approach is progressive and controllable, so intermediate samples can be "
                    "interpreted as a coarse-to-fine refinement of the final image.",
                    "On CIFAR-10 the model achieves an inception score of 9.46 and a Frechet inception "
                    "distance of 3.17, competitive with the best generative adversarial networks.",
                ],
            ),
            (
                "Comparison with adversarial training",
                [
                    "Diffusion sampling is slow because it needs many denoising steps, so distillation and "
                    "faster samplers are an active research area.",
                    "Unlike adversarial training the denoising objective is stable and does not suffer from "
                    "mode collapse, which is why it replaced generative adversarial networks for image work.",
                ],
            ),
        ],
    },
    {
        "key": "rlhf",
        "title": "Training Language Models to Follow Instructions",
        "year": 2022,
        "venue": "NeurIPS",
        "authors": "Ouyang et al.",
        "category": "alignment",
        "abstract": (
            "We fine-tune a large language model with human feedback so that it behaves as an "
            "instruction-following assistant. A reward model learned from human comparisons is "
            "optimised with reinforcement learning against a penalised objective."
        ),
        "sections": [
            (
                "Pipeline",
                [
                    "The pipeline has three stages: supervised fine-tuning on demonstrations, reward "
                    "model training on human preference comparisons, and policy optimisation with "
                    "proximal policy optimisation.",
                    "A per-token Kullback-Leibler penalty keeps the tuned policy close to the supervised "
                    "model and prevents reward hacking.",
                ],
            ),
            (
                "Findings",
                [
                    "Human raters prefer the tuned model over the base model, and the improvements "
                    "transfer to a held-out set of prompts that were never used for training.",
                    "The paper calls the mismatch between the training objective and human preferences "
                    "misalignment, and it is the reason evaluation on held-out prompts matters.",
                ],
            ),
            (
                "Evaluation",
                [
                    "Comparing two policies requires pairwise human judgement, and inter annotator "
                    "agreement is a real limit on reward model quality.",
                    "Automatic benchmarks saturate quickly, so held out prompts and red teaming are used "
                    "alongside preference metrics.",
                ],
            ),
        ],
    },
    {
        "key": "rag",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
        "year": 2020,
        "venue": "NeurIPS",
        "authors": "Lewis et al.",
        "category": "retrieval",
        "abstract": (
            "We combine a parametric sequence-to-sequence generator with a non-parametric dense "
            "retriever over a document index. The retriever supplies passages at generation time, "
            "so knowledge can be updated without retraining the model."
        ),
        "sections": [
            (
                "Method",
                [
                    "The retriever encodes the question and the documents separately with a dual "
                    "encoder, and the generator conditions on the retrieved passages to produce an "
                    "answer.",
                    "Because the retriever is non-parametric, the document index can be swapped for a "
                    "new one without changing a single generator weight.",
                ],
            ),
            (
                "Why it helps",
                [
                    "Conditioning on retrieved evidence reduces hallucination compared with a purely "
                    "parametric model, and the retrieved passage provides a citation a reader can check.",
                    "The approach is strongest on knowledge-intensive tasks where a closed-book model "
                    "cannot store all the required facts in its parameters.",
                ],
            ),
            (
                "Variants",
                [
                    "Later systems rerank retrieved passages with a cross-encoder before generation, which "
                    "raises precision when the retriever returns many candidates.",
                    "Retrieval can also be trained jointly with generation, so the gradients from the answer "
                    "flow back into the retriever instead of being frozen.",
                ],
            ),
        ],
    },
    {
        "key": "dpr",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "year": 2020,
        "venue": "EMNLP",
        "authors": "Karpukhin et al.",
        "category": "retrieval",
        "abstract": (
            "We show that retrieval can be implemented with a dual encoder over dense vectors and "
            "trained from question-passage pairs alone. This outperforms a strong sparse BM25 "
            "baseline on open-domain question answering."
        ),
        "sections": [
            (
                "Training",
                [
                    "The dual encoder is trained with an in-batch negative objective, using gold "
                    "passages as positives and passages retrieved by BM25 as hard negatives.",
                    "Similarity is measured with the dot product of the normalised question and passage "
                    "vectors, which makes retrieval a maximum inner product search problem.",
                ],
            ),
            (
                "Comparison",
                [
                    "Dense retrieval captures lexical mismatch that BM25 misses, for example when the "
                    "question and the passage share meaning but not words.",
                    "Hybrid strategies that combine sparse and dense scores are reported to give the "
                    "best accuracy of all in the paper.",
                ],
            ),
            (
                "Indexing",
                [
                    "The passage encoder runs offline over the whole corpus while the question encoder runs "
                    "online, so answering a question is a single vector search.",
                    "A FAISS index makes the maximum inner product search over millions of passages fast "
                    "enough for interactive question answering.",
                ],
            ),
        ],
    },
    {
        "key": "bm25",
        "title": "Okapi at TREC-3: Probabilistic Ranking",
        "year": 1994,
        "venue": "TREC",
        "authors": "Robertson et al.",
        "category": "retrieval",
        "abstract": (
            "We describe the Okapi probabilistic retrieval system and its ranking function, which "
            "remains the standard sparse baseline for text retrieval. The function combines term "
            "frequency saturation with document length normalisation."
        ),
        "sections": [
            (
                "Ranking function",
                [
                    "The score of a document is the sum over query terms of an inverse document "
                    "frequency weight multiplied by a saturated term frequency factor.",
                    "The parameter k1 controls how quickly term frequency saturates, and the parameter "
                    "b controls how strongly long documents are penalised.",
                    "Setting b to zero disables length normalisation, while the common default of 0.75 "
                    "is a compromise that works well across collections.",
                ],
            ),
            (
                "Practical notes",
                [
                    "Inverse document frequency is computed over the whole collection, so adding "
                    "documents changes the score of every existing document.",
                    "BM25 needs no training and no GPU, which is why it is still the first baseline a "
                    "retrieval system is measured against.",
                ],
            ),
            (
                "Extensions",
                [
                    "Field weighting and proximity boosts are common additions, and the fielded variant "
                    "applies the same saturation separately to each document field.",
                    "Query expansion with pseudo relevance feedback can raise recall at the cost of "
                    "additional latency and a second retrieval pass.",
                ],
            ),
        ],
    },
    {
        "key": "sbert",
        "title": "Sentence Embeddings using Siamese Networks",
        "year": 2019,
        "venue": "EMNLP",
        "authors": "Reimers and Gurevych",
        "category": "retrieval",
        "abstract": (
            "We derive sentence embeddings that are comparable with cosine similarity by fine-tuning "
            "a pretrained transformer on a siamese network. The resulting encoder is orders of "
            "magnitude faster than a cross-encoder at comparable accuracy."
        ),
        "sections": [
            (
                "Motivation",
                [
                    "A cross-encoder that feeds both sentences through the transformer jointly is "
                    "accurate but too slow to score millions of pairs at query time.",
                    "Pooling the token embeddings with mean pooling produces a fixed-size vector that "
                    "can be compared in constant time.",
                ],
            ),
            (
                "Training data",
                [
                    "The network is trained on natural language inference data, where entailed pairs "
                    "are positives and contradictory pairs are negatives, which makes the embedding "
                    "space meaningful for semantic similarity.",
                    "The same encoder transfers to clustering and semantic search with no further "
                    "training.",
                ],
            ),
            (
                "Limitations",
                [
                    "Mean pooling discards word order, so negation and other order sensitive phenomena are "
                    "represented imperfectly by a single sentence vector.",
                    "Training with hard negatives mined from a first pass gives much sharper embeddings "
                    "than sampling random negatives alone.",
                ],
            ),
        ],
    },
    {
        "key": "fedavg",
        "title": "Communication-Efficient Learning of Decentralised Data",
        "year": 2017,
        "venue": "AISTATS",
        "authors": "McMahan et al.",
        "category": "distributed systems",
        "abstract": (
            "We investigate federated learning, a setting where training data stays on mobile "
            "devices and only model updates are communicated. Averaging locally computed updates "
            "requires far fewer communication rounds than synchronised stochastic gradient descent."
        ),
        "sections": [
            (
                "Algorithm",
                [
                    "Each round samples a subset of clients, and every selected client performs several "
                    "epochs of local SGD on its own private data.",
                    "The server then averages the resulting client models, weighted by the number of "
                    "local examples, to form the next global model.",
                ],
            ),
            (
                "Trade-offs",
                [
                    "More local computation per round reduces the number of communication rounds, "
                    "which is usually the bottleneck on a mobile network.",
                    "The approach assumes clients are honest, and a malicious client can bias the "
                    "global model, which motivates robust aggregation research.",
                ],
            ),
            (
                "Extensions",
                [
                    "Secure aggregation and differential privacy protect individual updates from the "
                    "server, at some cost in accuracy and communication.",
                    "Personalisation adds a small per client adapter so that a single global model can "
                    "still fit heterogeneous devices.",
                ],
            ),
        ],
    },
    {
        "key": "vit",
        "title": "An Image Is Worth Sixteen By Sixteen Words",
        "year": 2021,
        "venue": "ICLR",
        "authors": "Dosovitskiy et al.",
        "category": "generative vision",
        "abstract": (
            "We show that a pure transformer applied directly to sequences of image patches performs "
            "very well on image classification. The architecture needs large-scale pretraining to "
            "compete with convolutional networks."
        ),
        "sections": [
            (
                "Architecture",
                [
                    "An image is split into fixed-size patches, each patch is flattened and linearly "
                    "projected, and the resulting sequence is fed to a standard transformer encoder.",
                    "A learnable class token is prepended, and its final representation is used as the "
                    "image representation for classification.",
                ],
            ),
            (
                "Inductive bias",
                [
                    "Transformers lack the locality and translation equivariance that convolutions "
                    "build in, so they generalise worse when trained on small datasets.",
                    "Pretraining on a very large dataset such as ImageNet-21k or JFT-300M removes that "
                    "disadvantage and the model then transfers well to smaller benchmarks.",
                ],
            ),
            (
                "Variants",
                [
                    "Hybrid models interleave convolutional stems with transformer blocks to reintroduce "
                    "locality at low cost.",
                    "Self supervised pretraining such as masked image modelling reduces the dependence on "
                    "labelled data and closes much of the remaining gap to convolutional networks.",
                ],
            ),
        ],
    },
    {
        "key": "lora",
        "title": "Low-Rank Adaptation of Large Language Models",
        "year": 2021,
        "venue": "ICLR",
        "authors": "Hu et al.",
        "category": "efficiency",
        "abstract": (
            "We freeze the pretrained weights and inject trainable low-rank matrices into each layer "
            "of the transformer. This reduces the number of trainable parameters by several orders "
            "of magnitude with no added inference latency."
        ),
        "sections": [
            (
                "Method",
                [
                    "For a frozen weight matrix the update is reparameterised as the product of two "
                    "much smaller matrices, so the number of trainable parameters drops sharply.",
                    "At inference time the low-rank matrices can be merged into the frozen weights, so "
                    "the adapted model has exactly the same architecture and latency as the base model.",
                ],
            ),
            (
                "Benefits",
                [
                    "Different tasks can be served from the same base model by swapping only the small "
                    "adapter weights, which makes multi-task serving practical.",
                    "The method is orthogonal to quantisation and pruning and is usually combined with "
                    "them in practice.",
                ],
            ),
            (
                "Configuration",
                [
                    "The rank and the choice of adapted weight matrices are the main knobs, and a small "
                    "rank applied to the attention projections is a common default.",
                    "Adapters can differ per client, which also makes them useful when a shared base "
                    "model has to serve many downstream tasks.",
                ],
            ),
        ],
    },
    {
        "key": "gcn",
        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
        "year": 2017,
        "venue": "ICLR",
        "authors": "Kipf and Welling",
        "category": "graphs",
        "abstract": (
            "We present a scalable approach for semi-supervised learning on graph-structured data "
            "based on an efficient variant of convolutional neural networks that operate directly "
            "on graphs."
        ),
        "sections": [
            (
                "Layer",
                [
                    "Each layer propagates features along graph edges and then applies a shared linear "
                    "transformation followed by a nonlinearity.",
                    "The propagation rule is a first-order approximation of a spectral graph "
                    "convolution, with a renormalisation trick that stabilises training on graphs with "
                    "high-degree nodes.",
                ],
            ),
            (
                "Results",
                [
                    "Using only a handful of labelled nodes per class, the model outperforms "
                    "graph regularisation baselines on citation network datasets.",
                    "The computational cost grows linearly in the number of edges, which is what makes "
                    "the method practical on large graphs.",
                ],
            ),
            (
                "Variants",
                [
                    "Graph attention networks learn a weight per edge instead of relying on fixed "
                    "normalisation, which helps when the graph is noisy.",
                    "Sampling based methods scale to graphs that do not fit in memory by training on "
                    "fixed size neighbourhoods rather than the whole graph.",
                ],
            ),
        ],
    },
    {
        "key": "efficient_attention",
        "title": "Efficient Transformers: Sparse and Linear Attention",
        "year": 2020,
        "venue": "ACM Computing Surveys",
        "authors": "Tay et al.",
        "category": "natural language processing",
        "abstract": (
            "We survey efficient attention variants that reduce the quadratic cost of self attention "
            "over long sequences. Approaches fall into fixed patterns, learned patterns, low rank "
            "approximations and memory-based methods."
        ),
        "sections": [
            (
                "Fixed patterns",
                [
                    "Block and stride patterns restrict each query to a fixed set of keys, which drops the "
                    "quadratic attention term to linear in the sequence length.",
                    "Sliding window attention keeps a local context for every token and combines layers so "
                    "that information propagates across the whole sequence.",
                ],
            ),
            (
                "Low rank and kernels",
                [
                    "Low rank approximations factorise the attention matrix, while kernel methods "
                    "rewrite attention as a linear operation over feature maps.",
                    "Both families trade exactness for throughput and tend to lose accuracy on tasks that "
                    "need precise long range retrieval of a single token.",
                ],
            ),
            (
                "Evaluation",
                [
                    "Comparisons are usually reported on long document classification and on retrieval of "
                    "a fact from a long context, where sparse attention degrades gracefully.",
                ],
            ),
        ],
    },
    {
        "key": "colbert",
        "title": "Late Interaction Retrieval over Token Embeddings",
        "year": 2020,
        "venue": "SIGIR",
        "authors": "Khattab and Zaharia",
        "category": "retrieval",
        "abstract": (
            "We propose a late interaction model that keeps one vector per token and scores a query "
            "passage pair with a cheap maximum similarity operation. It keeps much of the quality of a "
            "cross-encoder at a fraction of the cost."
        ),
        "sections": [
            (
                "Scoring",
                [
                    "Every query token is matched against every passage token and the maximum similarity "
                    "is summed, so the passage is scored jointly rather than as a single vector.",
                    "Because the token embeddings are precomputed, the interaction is a matrix operation "
                    "rather than a forward pass through a large encoder.",
                ],
            ),
            (
                "Comparison with cross-encoders",
                [
                    "A cross-encoder still reranks the top candidates more accurately, and production "
                    "systems commonly use late interaction to shortlist and a cross-encoder to order.",
                    "Late interaction needs far more index storage than a single passage vector, which is "
                    "the main practical cost of the approach.",
                ],
            ),
        ],
    },
    {
        "key": "prf",
        "title": "Query Expansion with Pseudo Relevance Feedback",
        "year": 1998,
        "venue": "SIGIR",
        "authors": "Robertson and Walker",
        "category": "retrieval",
        "abstract": (
            "We evaluate automatic query expansion, where the top ranked documents from a first pass "
            "are used to select additional query terms. Expansion raises recall but can also drift away "
            "from the original information need."
        ),
        "sections": [
            (
                "Method",
                [
                    "A first retrieval pass produces a candidate set and the most discriminating terms "
                    "from those documents are added to the query with a weight.",
                    "Relevance feedback from a human is more reliable than pseudo relevance feedback, which "
                    "assumes the first pass is mostly correct.",
                ],
            ),
            (
                "Effect on metrics",
                [
                    "Expansion usually improves recall at larger cut offs while precision at one can fall, "
                    "so the choice depends on whether a user reads the first result or scans a list.",
                    "Query drift is the failure mode: the expanded query retrieves a coherent but wrong "
                    "topic, and no automatic metric flags it reliably.",
                ],
            ),
        ],
    },
    {
        "key": "beir",
        "title": "A Heterogeneous Benchmark for Zero-Shot Retrieval Evaluation",
        "year": 2021,
        "venue": "NeurIPS Datasets",
        "authors": "Thakur et al.",
        "category": "retrieval",
        "abstract": (
            "We release a benchmark of diverse retrieval datasets to test whether a model trained on "
            "one collection transfers to another. Strong in-domain models turn out to generalise poorly "
            "without training on the target data."
        ),
        "sections": [
            (
                "Metrics",
                [
                    "Retrieval quality is reported with recall at several cut offs, mean reciprocal rank "
                    "and normalised discounted cumulative gain, because no single number captures both "
                    "ranking position and coverage.",
                    "Recall at k measures whether the evidence was retrieved at all, while reciprocal rank "
                    "measures how high it was ranked, and a production system needs both.",
                ],
            ),
            (
                "Findings",
                [
                    "BM25 is a remarkably strong baseline on out of domain collections, and hybrid "
                    "approaches that combine sparse and dense scores are the most robust across them.",
                    "Larger labeled training sets matter more than larger models for transfer, which is "
                    "why hard negatives and distillation are common in later work.",
                ],
            ),
        ],
    },
    {
        "key": "rag_survey",
        "title": "Retrieval-Augmented Generation: Methods and Open Problems",
        "year": 2023,
        "venue": "Survey",
        "authors": "Gao et al.",
        "category": "retrieval",
        "abstract": (
            "We organise recent retrieval augmented generation systems by retrieval stage, fusion "
            "strategy and generation recipe, and list the evaluation problems that remain open."
        ),
        "sections": [
            (
                "Architectures",
                [
                    "Systems differ in whether retrieval happens once before generation, at every decoding "
                    "step, or iteratively, with a critique model deciding when the evidence is sufficient.",
                    "Hybrid retrieval that fuses a sparse BM25 ranking with a dense FAISS ranking is the "
                    "most common default because it does not depend on the retriever being well trained.",
                ],
            ),
            (
                "Evaluation",
                [
                    "Retrieval is measured with recall at k and mean reciprocal rank on a labelled set, "
                    "while the final answer needs faithfulness and citation accuracy measured separately.",
                    "A common failure is a correct retriever paired with a generator that ignores the "
                    "evidence, which retrieval metrics cannot detect.",
                ],
            ),
        ],
    },
    {
        "key": "cfg",
        "title": "Classifier-Free Diffusion Guidance",
        "year": 2022,
        "venue": "NeurIPS Workshop",
        "authors": "Ho and Salimans",
        "category": "generative vision",
        "abstract": (
            "We show that a diffusion model can be guided towards a conditioning signal without training "
            "a separate classifier. The guidance weight trades sample diversity for sample fidelity."
        ),
        "sections": [
            (
                "Method",
                [
                    "The same network is trained with and without the condition by randomly dropping it, "
                    "and the two predictions are combined at sampling time.",
                    "Guidance requires two forward passes per denoising step, which roughly doubles the "
                    "sampling cost of an already slow iterative process.",
                ],
            ),
            (
                "Samplers",
                [
                    "Fewer denoising steps with a higher order sampler is the usual way to reduce sampling "
                    "cost, and distillation trains a student to imitate a multi step trajectory.",
                    "Both approaches trade sample quality for latency, and the acceptable operating point "
                    "depends on the application.",
                ],
            ),
        ],
    },
    {
        "key": "adapters",
        "title": "Parameter-Efficient Transfer Learning Compared",
        "year": 2022,
        "venue": "ACL",
        "authors": "Ruckle et al.",
        "category": "efficiency",
        "abstract": (
            "We compare adapter modules, prefix tuning and low rank reparameterisation under a matched "
            "parameter budget and find that the choice matters less than the budget itself."
        ),
        "sections": [
            (
                "Comparison",
                [
                    "With an equal number of trainable parameters the methods perform similarly, so the rank "
                    "and the placement of the adapted matrices matter more than the reparameterisation.",
                    "Prefix tuning keeps the base weights untouched but reserves context length, which can "
                    "be a real cost when the input is already long.",
                ],
            ),
            (
                "Serving",
                [
                    "All of the methods allow one base model to serve many tasks by loading a small set of "
                    "adapter weights, but only some can be merged so that inference latency is unchanged.",
                    "Merging is what makes a parameter efficient method attractive for latency sensitive "
                    "deployments, and it is not available for attention prefix methods.",
                ],
            ),
        ],
    },
    {
        "key": "hybrid_eval",
        "title": "On the Effectiveness of Hybrid Sparse and Dense Retrieval",
        "year": 2023,
        "venue": "ECIR",
        "authors": "Bruch et al.",
        "category": "retrieval",
        "abstract": (
            "We study when combining sparse and dense retrieval actually helps, and find that the gain "
            "depends on how differently the two scorers rank the same candidate pool."
        ),
        "sections": [
            (
                "Fusion",
                [
                    "Reciprocal rank fusion needs no score calibration and is competitive with tuned "
                    "weighted sums, which is why it is the usual default in hybrid systems.",
                    "Score normalisation before a weighted sum helps only when the two retrievers are "
                    "already individually well calibrated, otherwise fusion by rank is safer.",
                ],
            ),
            (
                "When it fails",
                [
                    "If both retrievers make the same mistake the fused ranking inherits it, so hybrid "
                    "search is not a fix for a corpus with no relevant passage at all.",
                    "Diversity between the two rankings, not the absolute quality of either one, is what "
                    "predicts whether fusion will improve recall at k.",
                ],
            ),
        ],
    },
]


# --------------------------------------------------------------------------
# Evaluation queries. ``fragment`` is matched against the corpus to recover the
# full anchor sentence, so a typo or a stale quote fails loudly instead of
# quietly producing an unresolvable label.
#
# Only the 12 "core" documents carry labels. The other 8 documents are
# deliberate distractors: they discuss the same topics with overlapping
# vocabulary, which is what makes a candidate pool of ~30 chunks a real test
# rather than a formality.
# --------------------------------------------------------------------------
QUERIES: list[dict] = [
    {"query": "Why does the Transformer need positional encodings?", "doc": "attention", "fragment": "no inherent notion of token order"},
    {"query": "What BLEU score does the big Transformer reach on English to German?", "doc": "attention", "fragment": "reaches 28.4 BLEU"},
    {"query": "What fraction of input tokens does masked language modelling mask?", "doc": "bert", "fragment": "randomly masks fifteen percent"},
    {"query": "How does the next sentence prediction objective hold up?", "doc": "bert", "fragment": "contributes little to downstream accuracy"},
    {"query": "What does the diffusion reverse process predict at each step?", "doc": "ddpm", "fragment": "trained to predict the noise that was added"},
    {"query": "What Frechet inception distance does the diffusion model reach on CIFAR-10?", "doc": "ddpm", "fragment": "Frechet inception distance of 3.17"},
    {"query": "What keeps the RLHF policy close to the supervised model?", "doc": "rlhf", "fragment": "Kullback-Leibler penalty"},
    {"query": "What are the three stages of the instruction tuning pipeline?", "doc": "rlhf", "fragment": "three stages: supervised fine-tuning"},
    {"query": "Can the RAG document index be replaced without retraining?", "doc": "rag", "fragment": "swapped for a new one without changing"},
    {"query": "How does conditioning on retrieved evidence reduce hallucination?", "doc": "rag", "fragment": "reduces hallucination compared with a purely"},
    {"query": "How is the dense passage retriever trained with hard negatives?", "doc": "dpr", "fragment": "in-batch negative objective"},
    {"query": "Do hybrid sparse and dense scores beat either alone?", "doc": "dpr", "fragment": "best accuracy of all"},
    {"query": "What does the k1 parameter control in Okapi BM25?", "doc": "bm25", "fragment": "controls how quickly term frequency saturates"},
    {"query": "What does setting b to zero do in BM25?", "doc": "bm25", "fragment": "disables length normalisation"},
    {"query": "Why is a cross-encoder too slow for retrieval?", "doc": "sbert", "fragment": "too slow to score millions of pairs"},
    {"query": "What training data makes sentence embedding spaces meaningful?", "doc": "sbert", "fragment": "entailed pairs"},
    {"query": "How are client models combined in federated averaging?", "doc": "fedavg", "fragment": "averages the resulting client models"},
    {"query": "What is the bottleneck on a mobile network in federated learning?", "doc": "fedavg", "fragment": "number of communication rounds"},
    {"query": "What inductive bias do vision transformers lack?", "doc": "vit", "fragment": "locality and translation equivariance"},
    {"query": "How large does the pretraining dataset need to be for a vision transformer?", "doc": "vit", "fragment": "ImageNet-21k or JFT-300M"},
    {"query": "Does LoRA add inference latency?", "doc": "lora", "fragment": "same architecture and latency"},
    {"query": "How does low-rank adaptation reduce trainable parameters?", "doc": "lora", "fragment": "number of trainable parameters drops sharply"},
    {"query": "How does the graph convolution renormalisation help training?", "doc": "gcn", "fragment": "stabilises training on graphs"},
    {"query": "How does the cost of a graph convolutional layer scale?", "doc": "gcn", "fragment": "linearly in the number of edges"},
    # Second pass: one query into each document's follow-up sections, so the
    # label set spans the whole corpus rather than only the opening paragraph.
    {"query": "What is the cost of self-attention in the sequence length?", "doc": "attention", "fragment": "quadratic time in the sequence length"},
    {"query": "How much of the input does each masked pretraining step supervise?", "doc": "bert", "fragment": "supervises fifteen percent of the input tokens"},
    {"query": "Why is diffusion sampling slow?", "doc": "ddpm", "fragment": "needs many denoising steps"},
    {"query": "What limits reward model quality in practice?", "doc": "rlhf", "fragment": "inter annotator agreement"},
    {"query": "What does reranking retrieved passages do before generation?", "doc": "rag", "fragment": "raises precision when the retriever returns many candidates"},
    {"query": "Why does a FAISS index matter for open-domain question answering?", "doc": "dpr", "fragment": "fast enough for interactive question answering"},
    {"query": "How can pseudo relevance feedback change recall in BM25?", "doc": "bm25", "fragment": "can raise recall at the cost of additional latency"},
    {"query": "What does mean pooling lose in a sentence embedding?", "doc": "sbert", "fragment": "discards word order"},
    {"query": "What protects individual client updates from the server?", "doc": "fedavg", "fragment": "Secure aggregation and differential privacy"},
    {"query": "How do hybrid vision models reintroduce locality?", "doc": "vit", "fragment": "interleave convolutional stems with transformer blocks"},
    {"query": "Which weight matrices does low-rank adaptation usually adapt?", "doc": "lora", "fragment": "small rank applied to the attention projections"},
    {"query": "How do graph attention networks differ from fixed normalisation?", "doc": "gcn", "fragment": "learn a weight per edge"},
    # Coverage queries: two gold passages each, deliberately spanning documents
    # (including the distractor ones) so Recall@K measures coverage rather than
    # "did the single right answer come back".
    {
        "query": "How do sparse and dense retrieval scores differ in what they capture?",
        "fragments": [
            ("bm25", "inverse document frequency weight multiplied"),
            ("dpr", "dot product of the normalised question and passage"),
        ],
    },
    {
        "query": "What two rankings does a hybrid retriever fuse?",
        "fragments": [
            ("hybrid_eval", "Reciprocal rank fusion needs no score calibration"),
            ("rag_survey", "fuses a sparse BM25 ranking"),
        ],
    },
    {
        "query": "Which metrics are used to evaluate retrieval and the final answer?",
        "fragments": [
            ("beir", "recall at several cut offs, mean reciprocal rank"),
            ("rag_survey", "faithfulness and citation accuracy measured separately"),
        ],
    },
    {
        "query": "How does a cross-encoder differ from a bi-encoder?",
        "fragments": [
            ("sbert", "feeds both sentences through the transformer jointly"),
            ("colbert", "A cross-encoder still reranks the top candidates"),
        ],
    },
    {
        "query": "How can diffusion sampling cost be reduced?",
        "fragments": [
            ("cfg", "Fewer denoising steps with a higher order sampler"),
            ("ddpm", "distillation and faster samplers are an active research area"),
        ],
    },
    {
        "query": "Which parameter efficient methods keep inference latency unchanged?",
        "fragments": [
            ("adapters", "only some can be merged so that inference latency is unchanged"),
            ("lora", "merged into the frozen weights"),
        ],
    },
]

# Two queries carry a filter, to exercise metadata-filtered retrieval in the
# evaluation rather than only in unit tests.
FILTERED = {
    "Do hybrid sparse and dense scores beat either alone?": {"category": "retrieval"},
    "What keeps the RLHF policy close to the supervised model?": ["year >= 2020", "category = alignment"],
    "What does reranking retrieved passages do before generation?": {"category": "retrieval"},
    "Why is diffusion sampling slow?": ["year >= 2020", "category = generative vision"],
}


def document_body(doc: dict) -> str:
    """Render a document's sections into the text that gets chunked."""
    parts = [doc["abstract"]]
    for heading, sentences in doc["sections"]:
        parts.append(f"{heading}\n" + " ".join(sentences))
    return "\n\n".join(parts)


def all_sentences(doc: dict) -> list[str]:
    sentences = [doc["abstract"]]
    for _heading, body in doc["sections"]:
        sentences.extend(body)
    return sentences


def resolve_anchor(doc: dict, fragment: str) -> str:
    """Return the single sentence containing ``fragment``, or fail loudly."""
    matches = [sentence for sentence in all_sentences(doc) if fragment in sentence]
    if len(matches) != 1:
        raise SystemExit(
            f"anchor {fragment!r} in {doc['key']!r} matched {len(matches)} sentences; "
            "make the fragment unique"
        )
    return matches[0]


def write_corpus(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    for doc in DOCS:
        body = document_body(doc)
        path = directory / f"{doc['key']}_{doc['year']}.txt"
        # A leading title line, then the body. The metadata sidecar carries the
        # structured fields; the filename year is a fallback the ingest layer
        # also understands.
        path.write_text(f"{doc['title']}\n\n{body}\n", encoding="utf-8")
        meta = {
            "title": doc["title"],
            "year": doc["year"],
            "venue": doc["venue"],
            "authors": doc["authors"],
            "category": doc["category"],
            "key": doc["key"],
            "synthetic": True,
        }
        (directory / f"{path.name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return len(DOCS)


def write_testset(path: Path) -> int:
    by_key = {doc["key"]: doc for doc in DOCS}
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for item in QUERIES:
        if "fragments" in item:
            # A coverage query: several passages are needed to answer it. These
            # are the queries that make Recall@K differ from hit rate - a single
            # gold passage per query makes recall@k and hit@k the same number.
            pairs = list(item["fragments"])
            record = {
                "query": item["query"],
                "anchors": [resolve_anchor(by_key[key], fragment) for key, fragment in pairs],
                "doc_hints": [key for key, _ in pairs],
            }
        else:
            doc = by_key[item["doc"]]
            record = {
                "query": item["query"],
                "anchor": resolve_anchor(doc, item["fragment"]),
                "doc_hint": doc["key"],
            }
        if item["query"] in FILTERED:
            record["filters"] = FILTERED[item["query"]]
        lines.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def write_pdfs(directory: Path) -> int:
    """Emit real PDFs so the PyPDF ingestion path can be exercised end to end."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("reportlab is not installed; skipping PDF generation (pip install reportlab)")
        return 0

    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for doc in DOCS:
        path = directory / f"{doc['key']}_{doc['year']}.pdf"
        pdf = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        # Two pages: metadata page one, body page two. Multi-page extraction is
        # what makes page-number citations meaningful, so the sample exercises it.
        pdf.setFont("Helvetica-Bold", 15)
        pdf.drawString(60, height - 70, doc["title"])
        pdf.setFont("Helvetica", 10)
        pdf.drawString(60, height - 90, f"{doc['authors']} - {doc['venue']} {doc['year']}")
        pdf.drawString(60, height - 106, f"category: {doc['category']}")
        pdf.setFont("Helvetica", 11)
        y = height - 140
        for line in _wrap(doc["abstract"], 90):
            pdf.drawString(60, y, line)
            y -= 15
        pdf.showPage()
        y = height - 70
        for heading, sentences in doc["sections"]:
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(60, y, heading)
            y -= 18
            pdf.setFont("Helvetica", 11)
            for line in _wrap(" ".join(sentences), 90):
                if y < 60:
                    pdf.showPage()
                    y = height - 70
                    pdf.setFont("Helvetica", 11)
                pdf.drawString(60, y, line)
                y -= 15
            y -= 10
        pdf.save()
        written += 1
    return written


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="also write real PDFs for the pypdf path")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--testset", type=Path, default=TESTSET)
    args = parser.parse_args(argv)

    documents = write_corpus(args.corpus_dir)
    queries = write_testset(args.testset)
    print(f"wrote {documents} documents to {args.corpus_dir}")
    print(f"wrote {queries} labelled queries to {args.testset}")
    if args.pdf:
        pages = write_pdfs(PDF_DIR)
        print(f"wrote {pages} PDFs to {PDF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
