# Course Consolidation Plan

## Context

Consolidating a 20-session ML/AI weather/climate course. Outputs: README.md (lecture notes) and notebooks/scratch.ipynb (code). Original materials in `e-ai_ml2/` submodule. Sessions 1-6 are solid. This plan covers the remaining gaps, ordered by value, with specific content to add based on a thorough review of every course notebook.

---

## Tier 1 — High value (5 topics)

### 1. Diffusion — flexible graph networks

**Course ch. 9 / README "Diffusion and Flexible Graph Networks" (~line 991)**

Skim: `code09/5_Flex_Graph.ipynb`, `code09/5_Graph_flex_da.ipynb`, `code09/6_Graph_flex_Derivative.ipynb`

**Diffusion notes (minor additions to existing):**
- Closed-form forward: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
- Reverse: NN predicts noise $\epsilon$ given noisy sample + timestep embedding; loss = MSE(predicted, actual noise)
- Class-conditioned generation via one-hot label concatenation

**Flexible graph networks (flesh out ~line 1025):**
- Graph representation: nodes = grid points with features [value, obs_mask]; edges = k-nearest neighbours + self-loops
- Message-passing layer: $m_i = \sum_{j \in \mathcal{N}(i)} h_j$, then $h_i' = \text{ReLU}(W_\text{self} h_i + W_\text{neigh} m_i)$
- Position-agnostic learning: train on [0,π] N=30, generalise to [0,2π] N=60 with same weights — k-neighbour topology is scale-invariant
- ~17k params for hidden=64, 2 layers

**Graph-based DA (new subsection replacing "Graph structure exploration"):**
- Same GNN handles irregular obs→grid connections; learns to infer unobserved values via latent message passing

**Operator learning (new subsection):**
- Learn d²f/dx² from sparse f observations; same architecture, generalises to longer intervals/higher frequencies

**Optional code:** 2-layer GNN with edge_index construction and index_add_ aggregation in scratch.ipynb

---

### 2. AI Data Assimilation — AI-VAR and particle filters

**Course ch. 18 / README "AI Data Assimilation" (~line 1337)**

Skim: `code18/1_Inversion_1D.ipynb`, `code18/2_AI-VAR_1d.ipynb`, `code18/4_Lorenz63_DA_AIPF.ipynb`, `code18/5_Lorenz63_DA_AIPF_GM.ipynb`

**Introduction to AI-based DA (flesh out ~line 1339):**
- 3D-VAR cost function: $J = \frac{1}{2}(x-x_b)^T B^{-1}(x-x_b) + \frac{1}{2}(y-Hx)^T R^{-1}(y-Hx)$
- Closed-form analysis: $x_a = x_b + BH^T(HBH^T + R)^{-1}(y - Hx_b)$
- Course notebook 1 uses modulated sine truth on 256-point periodic grid with sparse obs

**AI-VAR (flesh out ~line 1343):**
- IncrementMLP: input [x_b, y_grid, mask] (3n dims) → 256 → 256 → n (Tanh activations) → output increment δx
- x_a = x_b + δx; trained to minimise J directly (same cost function as 3D-VAR)
- Comparison table:

| | Traditional 3D-VAR | AI-VAR |
|---|---|---|
| B matrix | Explicit (Gaussian covariance) | Learned implicitly by network |
| TL/AD | Required (dM/dx, M^T) | Not needed |
| Runtime | Solve linear system O(n³) | Single forward pass O(n) |
| Analysis | x_a = x_b + BH^T S^{-1} d | x_a = x_b + NN(x_b, y, mask) |

**AI ensemble DA (flesh out ~line 1356):**
- AI particle filter (AIPF): DeepSets architecture (φ→pool→ψ) for permutation-invariant ensemble updates on Lorenz63
    - φ: maps each particle + obs → embedding; ρ: average pool over particles; ψ: per-particle update using pooled context
    - Loss: SMC-style importance-weighted log-likelihood: $\mathcal{L} = -\log\left(\frac{1}{N}\sum_i \exp(\log p(y|x_a^i))\right)$
- Gaussian mixture background term: KL(target ‖ model) regularisation prevents filter collapse
    - Target: Kalman-updated particles weighted by likelihood; model: mixture over analysed particles

**Optional code:** Extend existing scratch modulated-sine setup with observation generation + IncrementMLP

---

### 3. RAG — pipeline and FAISS weather example

**Course ch. 7 / README "Retrieval-Augmented Generation" (~line 924)**

Skim: `code07/1_vector_db_elementary.ipynb`, `code07/5_icon_faiss_rag.ipynb`

**Document preparation (~line 937):**
- Chunking: line-based splitting with overlap (e.g. 1800 chars, 300 overlap); skip binaries and files >2MB
- Metadata per chunk: filename, line range, chunk ID — enables traceability to source

**Embedding generation (~line 941):**
- SentenceTransformer ("all-MiniLM-L6-v2"): maps text → 384-dim vectors, normalised to unit norm
- Same embedding space for documents and queries; cosine similarity = inner product on unit vectors

**Vector databases, chunking and persistence (~line 949):**
- FAISS IndexFlatIP: exact inner-product search on normalised vectors
- Scales to tens of thousands of chunks; C++ backend
- Similarity scores: ~0.7 = related topic, relative ranking more important than absolute

**End-to-end RAG pipelines (~line 953):**
- Context assembly: retrieve top-K chunks, concatenate with references (max ~12k chars)
- System prompt: "answer strictly based on provided context; include sources section with filenames + line ranges"
- Weather example: ICON model repo indexed (1211 files → 26,883 chunks); query "How is cloud microphysics treated in ICON?" retrieves relevant Fortran modules with similarity ~0.63

**Local and hosted LLMs (~line 943):**
- Same interface works with OpenAI (gpt-4o-mini) or local Ollama (llama3); provider flexibility
- Local LLMs preserve data sovereignty

---

### 4. Physics-informed learning — PINNs, SINDy, constraints, causality

**Course ch. 19 / README "AI and Physics" (~line 1360)**

Skim: `code19/01_PINN_sine.ipynb`, `code19/03_Lorenz63_SINDy.ipynb`, `code19/4_RHS_Learning_L63.ipynb`, `code19/5_Physical_Contraint.ipynb`, `code19/6_Causal_Modelling.ipynb`

**Extracting governing equations — SINDy (~line 1362):**
- Build library of candidate terms: $\Theta(X) = [1, x, y, z, xy, xz, yz, x^2, y^2, z^2, ...]$
- Regression: $\dot{X} = \Theta(X) \cdot \Xi$ where $\Xi$ is sparse coefficient matrix
- Sequential thresholded least-squares (STLSQ): zero coefficients below λ, re-fit active terms, repeat
- Discovers Lorenz-63 equations from trajectory data without assuming form
- Advantages over NNs: interpretable equations, few parameters, physics-agnostic discovery
- Limitations: must choose library a priori; noise amplifies in finite-difference derivatives

**Physics-informed neural networks (~line 1366):**
- PINN loss: $\mathcal{L} = w_\text{ODE} \cdot \frac{1}{N_c}\sum|r(x_i)|^2 + w_\text{BC} \cdot \sum|B(x_j)|^2$
    - $r(x_i)$: PDE residual at stochastic collocation points (e.g. $y'' + y$ for harmonic oscillator)
    - $B(x_j)$: boundary conditions at fixed points
- Derivatives computed via `torch.autograd.grad(..., create_graph=True)` — arbitrary order
- No training data needed: network learns by minimising physics violations
- Fourier feature embedding $\phi(x) = [\sin(\omega_1 x), \cos(\omega_1 x), ...]$ improves convergence for periodic solutions
- Domain separation: collocation domain can extend beyond problem domain for better extrapolation

**Neural RHS learning (brief note under PINNs or new bullet):**
- Learn dx/dt = f(x) from trajectory data
- Key: training over full state space (not just observed trajectory) needed for generalisation

**Embedding conservation laws (~line 1370):**
- Example: 1D periodic advection ($u_t + cu_x = 0$, $\int u \, dx = \text{const}$)
- Unconstrained MLP: fast but violates mass conservation (drift accumulates)
- Global constraint (zero-mean update): fixes conservation but allows high-frequency noise
- CNN with circular padding: naturally respects locality → better accuracy AND conservation
- Lesson: physical constraints alone insufficient; must pair with appropriate architecture (locality, smoothness)

**Causal modelling (new subsection after conservation):**
- Correlation ≠ causation; confounders create spurious links (e.g. forcing F drives both P and T independently)
- Naive regression P→T gives spurious coefficient; multiple regression controlling for F reveals truth
- PCMCI (Peter-Clark Momentary Conditional Independence): tests $X(t-\tau) \perp Y(t) | \text{parents of } Y$
    - Discovers correct causal graph in both direct-coupling and confounded scenarios

**Optional code:** PINN for y'' + y = 0 is a natural scratch.ipynb exercise (autograd chain, residual loss + BC loss)

---

### 5. Anemoi — graph construction and training pipeline

**Course ch. 15 / README "Anemoi: AI-Based Weather Modelling" (~line 1266)**

Skim: `code15/4_ERA_T2m_to_ZARR.ipynb`, `code15/5_Simple_Anemoi_Graph.ipynb`, `code15/6_Anemoi_Training.ipynb`

**Zarr and ERA datasets (flesh out ~line 1284):**
- NetCDF → Zarr conversion; chunked storage (e.g. 4×226×450 for time×lat×lon)
- Lazy loading via dask; `xr.open_zarr(..., consolidated=True)` for fast access
- Chunking trade-off: more chunks = more metadata overhead, fewer = larger loads
- Data cleaning: filter pole duplicates (lat=±90°), mask NaN ocean cells, normalise lon to [-180,180)

**Icosahedral graph construction (~line 1290):**
- Subdivide regular icosahedron N times → globally uniform node distribution (no pole singularities)
- Resolution 3 = 642 nodes; resolution 4 = 2,562 nodes
- KNN edges (k=6): 15,372 directed edges at res 4; uniformly distributed, no directional bias
- Node attributes: area weights (for loss weighting on sphere); edge attributes: length, direction
- Coordinates stored in radians; convert to degrees for interpolation

**Training and validation / End-to-end training pipelines (~line 1294+):**
- Interpolation: `scipy.interpolate.RegularGridInterpolator` maps ERA5 lat-lon grid to icosahedral nodes
- Autoregressive forecasting: input state at t → predict state at t+1; X = data[:-1], Y = data[1:]
- Model options (Hydra-selectable): GNN (message passing), GraphTransformer (attention), hierarchical (multi-scale encoder-processor-decoder with skip connections)
- Hierarchical variant: three resolution levels; processes coarse→intermediate→fine
- Config: `python train.py model=gnn training.epochs=200` etc.

---

## Tier 2 — Useful for breadth (3 topics)

### 6. Agents — function calling and LangGraph forecast

**Course ch. 10 / README "Agents and Coding with LLMs" (~line 1049)**

Skim: `code10/1_function_calling_basics.ipynb`, `code10/7_langgraph_get_forecast.ipynb`

**Automated coding with LLMs (~line 1053):**
- Function calling pattern: (1) define tool schema as JSON, (2) LLM decides whether to call, (3) system executes function, (4) result fed back as ToolMessage with matching tool_call_id
- LLM never executes code directly; separation of reasoning from execution
- Agent loop: cyclical decide→act→observe; errors/tracebacks fed back as context for correction; max-attempts guard

**LangGraph-based forecast assistant (~line 1063):**
- DAWID agent: LangGraph StateGraph with typed state dict flowing through linear pipeline
- 7 nodes: extract forecast datetime → get latest DWD forecast → calculate lead time → extract location → extract variable → download ICON-EU GRIB → plot with Cartopy
- Each node: pure function (state) → state; graph compiled and invoked with initial state
- Grounds LLM reasoning in live weather data (DWD open data, Nominatim geolocation)

---

### 7. Multimodal LLMs — ViT encoder and weather applications

**Course ch. 8 / README "Multimodal Large Language Models" (~line 956)**

Skim: `code08/2_multimodal_image_embedding_ViT.ipynb`, `code08/3_radar_composite.ipynb`, `code08/4_cth_interpretation.ipynb`

**Fundamentals (extend existing ~line 958):**
- Concrete architecture: frozen ViT-Base (768-dim CLS token) → trainable linear bridge (768→512) → T5-Small decoder
- Bridge layer is sole trainable cross-modal link; forces meaningful compression
- 12 epochs on 80 synthetic wind-field samples; loss 3.44 → 0.042

**Radar data access and interpretation (~line 983):**
- DWD radar composites: H5 format; reflectivity = raw × gain + offset (dBZ)
- dBZ ranges: light 5-15, moderate 15-25, heavy 25-35+; stratiform (uniform) vs convective (isolated cells)
- Vision-language models (GPT-4V) interpret mesoscale precipitation patterns from rendered images

**Cloud-top height as multimodal AI application (~line 987):**
- DWD satellite-derived CTH: NetCDF, 934×1601 grid; low CTH → deep convection
- Vision-language models map spatial CTH variations to synoptic features (fronts, instability)

---

### 8. Learning from Observations — ORIGEN concept

**Course ch. 20 / README "Learning from Observations Only" (~line 1374)**

Skim: `code20/2_learn_obs_to_obs.ipynb`, `code20/3_origen_oscillator.ipynb`, `code20/4_origen_L63.ipynb`

**Direct observation prediction (~line 1376):**
- Obs-to-obs mapping: NN predicts (SAT(t+1), RS(t+1)) from (SAT(t), RS(t))
- Architecture: CNN encoder on satellite curtains + DeepSets encoder for radiosonde points; RS query decoder samples local SAT features at query locations
- Can generalise to unseen radiosonde locations; can reconstruct full 2D field by evaluating at all grid points

**ORIGEN iterative model and DA learning (~line 1380):**
- Two-phase approach:
    1. Iterative 3D-Var: many cycles with random partial obs patterns; B → P^a covariance adaptation each cycle; produces increasingly accurate assimilated trajectory
    2. Train NN on final assimilated trajectory to learn one-step transition x(k) → x(k+1)
- Rollout-aware fine-tuning (multi-step loss, weight α=0.3) prevents learned dynamics from diverging
- No forward model needed; learns purely from assimilated observations
- Demonstrated on Lorenz-63 with partial variable obs and wrong model parameters

---

## Tier 3 — Skip unless interested

- **MLflow (ch. 12):** Tooling — log metrics/params/artefacts, model registry. MLOps notes already cover context.
- **DAWID (ch. 11):** DWD-specific. Front detection with CNNs is interesting but institution-specific.
- **CI/CD (ch. 14):** Zarr notebooks (4-7) more interesting than CI/CD mechanics. MLOps notes cover concepts.
- **AI Transformation (ch. 16) / Model Emulation (ch. 17):** Light conceptual chapters. Low consolidation value.

---

## Approach

For each topic above:

1. Skim the listed notebooks (focus on concepts, not every code cell)
2. Add concise notes to README.md under the existing section headings (content suggestions above)
3. Optionally re-implement the core idea in scratch.ipynb where noted

Work Tier 1 topics 1-5 in order. Then Tier 2 topics 6-8 for breadth.

## Verification

- `git diff README.md` after each topic to review changes
- Run pymarkdown linter: `pre-commit run pymarkdown --files README.md` (lines <192 chars, 4-space indent for nested lists)
- Verify course material paths: `ls e-ai_ml2/course/code/code{NN}/` for each chapter number
