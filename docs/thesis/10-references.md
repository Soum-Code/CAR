# References

Generated from [`references.bib`](references.bib) by
`python scripts/check_citations.py --markdown`. Do not edit by hand.

All entries were verified against the arXiv listing on 2026-09-07.
Any that regain an **[unverified]** marker have a `note` saying so and
must be checked before submission.

## Conformal prediction and risk control

- `angelopoulos2024crc` — **Conformal Risk Control** · Angelopoulos, Anastasios N. and Bates, Stephen and Fisch, Adam and Lei, Lihua and Schuster, Tal, International Conference on Learning Representations (ICLR), 2024 · [2208.02814](https://arxiv.org/abs/2208.02814)
- `vovk2005alrw` — **Algorithmic Learning in a Random World** · Vovk, Vladimir and Gammerman, Alexander and Shafer, Glenn, Springer, 2005
  <br>The split-conformal construction and its exchangeability assumption
- `gibbs2021aci` — **Adaptive Conformal Inference Under Distribution Shift** · Gibbs, Isaac and Candès, Emmanuel, Advances in Neural Information Processing Systems (NeurIPS), 2021 · [2106.00170](https://arxiv.org/abs/2106.00170)
  <br>The online threshold update this project's adaptive calibrator is built on
- `barber2023beyond` — **Conformal Prediction Beyond Exchangeability** · Barber, Rina Foygel and Candès, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J., The Annals of Statistics, 2023 · [2202.13415](https://arxiv.org/abs/2202.13415)
  <br>Weighted quantiles for distribution drift plus a randomization technique for nonsymmetric fitting algorithms. Handles EXOGENOUS violations of exchangeability; the one in this thesis is endogenous, so this is the closest starting point rather than a solution. See ch. 2.1 and ch. 9
- `khosravi2026csa` — **Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs** · Khosravi, Hamed and Huo, Xiaoming, 2026 · [2605.20270](https://arxiv.org/abs/2605.20270)
  <br>Theorem E.1 is the sparse-verifier result that supersedes this project's censored-feedback claim
- `kotte2026certify` — **When Can Conformal Risk Control Certify LLM Outputs? Bounds, Impossibility, and Adaptation for Structured Generation** · Kotte, Varun, 2026 · [2606.29054](https://arxiv.org/abs/2606.29054)
  <br>Proposition 3 is the impossibility bound used throughout as a feasibility instrument

## Benchmarks and step-labelled data

- `cobbe2021gsm8k` — **Training Verifiers to Solve Math Word Problems** · Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John, 2021 · [2110.14168](https://arxiv.org/abs/2110.14168)
  <br>GSM8K. The inline calculator annotations are what make the local validity check deterministic
- `geva2021strategyqa` — **Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies** · Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan, Transactions of the Association for Computational Linguistics, 2021 · [link](https://allenai.org/data/strategyqa)
  <br>Ch. 8 shows its annotated decompositions are too shallow to exhibit propagation
- `wang2024mathshepherd` — **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations** · Wang, Peiyi and Li, Lei and Shao, Zhihong and Xu, Runxin and Dai, Damai and Li, Yifei and Chen, Deli and Wu, Yu and Sui, Zhifang, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024 · [2312.08935](https://arxiv.org/abs/2312.08935)
  <br>The 93,129 labelled steps this thesis measures on. Labels are Monte-Carlo hard estimation, not human annotation

## Process supervision and step-level scoring

- `lightman2023verify` — **Let's Verify Step by Step** · Lightman, Hunter and Kosaraju, Vineet and Burda, Yura and Edwards, Harri and Baker, Bowen and Lee, Teddy and Leike, Jan and Schulman, John and Sutskever, Ilya and Cobbe, Karl, 2023 · [2305.20050](https://arxiv.org/abs/2305.20050)
  <br>Process supervision beats outcome supervision on mathematical reasoning
- `ni2025reprobe` — **ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models** · Ni, Jingwei and Fadeeva, Ekaterina and Wu, Tianyi and Akhtar, Mubashara and Zhang, Jiaheng and Ash, Elliott and Leippold, Markus and Baldwin, Timothy and Ng, See-Kiong and Shelmanov, Artem and Sachan, Mrinmaya, 2025 · [2511.06209](https://arxiv.org/abs/2511.06209)
  <br>A small probe on frozen internal states matching far larger PRMs; the efficiency result any "better step score" proposal must beat, and tested here in §7.6 across training data, target and functional form, none of which closes the gap to it
- `wen2026embedding` — **Embedding Perturbation may Better Reflect Intermediate-Step Uncertainty in LLM Reasoning** · Wen, Qihao and Wang, Jiahao and Nan, Yang and He, Pengfei and Tandon, Ravi and Xu, Han, 2026 · [2602.02427](https://arxiv.org/abs/2602.02427)
  <br>Argues embedding perturbation reflects intermediate-step uncertainty better than sampling-based signals do. Ch. 7 measures the sampling-based half and finds it near chance, which is the same conclusion from the other direction
- `farquhar2024semantic` — **Detecting Hallucinations in Large Language Models Using Semantic Entropy** · Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin, Nature, 2024
  <br>Whole-answer semantic entropy via bidirectional entailment clustering; ch. 7 applies the principle to one step and finds it does not carry

## Verification, self-correction and propagation

- `huang2024selfcorrect` — **Large Language Models Cannot Self-Correct Reasoning Yet** · Huang, Jie and Chen, Xinyun and Mishra, Swaroop and Zheng, Huaixiu Steven and Yu, Adams Wei and Song, Xinying and Zhou, Denny, International Conference on Learning Representations (ICLR), 2024 · [2310.01798](https://arxiv.org/abs/2310.01798)
  <br>The prediction ch. 5's same-model critic arm confirms at step level: scope 0.0000
- `singh2026snowball` — **The Hallucination Snowball: Modeling Error Propagation as State Transitions in Multi-Agent LLM Pipelines** · Singh, Prabhjot and Pawar, Bhushan, 2026 · [2608.14588](https://arxiv.org/abs/2608.14588)
  <br>Escape probabilities 24.6/48.3/89.3\% across successive boundaries; the source of this project's decay constant 0.377
- `ro2025sherlock` — **Sherlock: Reliable and Efficient Agentic Workflow Execution** · Ro, Yeonju and Qiu, Haoran and Goiri, 'Iñigo and Fonseca, Rodrigo and Bianchini, Ricardo and Akella, Aditya and Wang, Zhangyang and Erez, Mattan and Choukse, Esha, 2025 · [2511.00330](https://arxiv.org/abs/2511.00330)
  <br>Closest prior work to ch. 6: verifier placement on a known workflow DAG, chosen offline by structure
- `you2025probabilistic` — **Probabilistic Soundness Guarantees in LLM Reasoning Chains** · You, Weiqiu and Xue, Anton and Havaldar, Shreya and Rao, Delip and Jin, Helen and Callison-Burch, Chris and Wong, Eric, 2025 · [2507.12948](https://arxiv.org/abs/2507.12948)
  <br>Evaluates a step given verified premises, which is the local/global decomposition of ch. 3 approached from the other side

## Selective labels

- `lakkaraju2017selective` — **The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables** · Lakkaraju, Himabindu and Kleinberg, Jon and Leskovec, Jure and Ludwig, Jens and Mullainathan, Sendhil, Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017 · [link](https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf)
  <br>Why labels observed only where a policy acted cannot be used naively; the reason forced exploration is in the loop

## Models used in the measurements

- `jiang2023mistral` — **Mistral 7B** · Jiang, Albert Q. and Sablayrolles, Alexandre and Mensch, Arthur and Bamford, Chris and Chaplot, Devendra Singh and de las Casas, Diego and Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier, Lucile and others, 2023 · [2310.06825](https://arxiv.org/abs/2310.06825)
  <br>The base of Mistral-7B-SFT, the generator behind Math-Shepherd's GSM8K solutions
- `qwen2024qwen25` — **Qwen2.5 Technical Report** · Qwen Team, 2024 · [2412.15115](https://arxiv.org/abs/2412.15115)
  <br>Qwen2.5-7B-Instruct is the second generator (ch. 4) and the independent judge (ch. 5)
- `dubey2024llama3` — **The Llama 3 Herd of Models** · Llama Team, AI @ Meta, 2024 · [2407.21783](https://arxiv.org/abs/2407.21783)
  <br>The generator this thesis names but could not run: licence-gated on the available compute platform, see ch. 4 limits

