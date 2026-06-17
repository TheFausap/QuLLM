# Quantum-Native Encoding Research Agenda

## Framing

The core idea is not merely "make transformer weights complex." The stronger version is:

> Language enters the model as a normalized complex state, and the model's contextualization is primarily a sequence of unitary transformations plus explicit measurement/readout.

That changes the role of the encoder/tokenizer. Instead of token ID -> real vector, the input side should produce a state in a Hilbert space whose geometry already carries linguistic structure.

## Relevant Prior Work

- The original transformer defines attention as softmax-normalized query/key/value interaction over real-valued token embeddings. That is the classical baseline this project is departing from. Source: [Attention Is All You Need](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need).
- Quantum NLP via DisCoCat treats grammar and meaning composition as quantum-process-like diagrams, making syntax central rather than treating tokens as a flat sequence. Source: [Foundations for Near-Term Quantum Natural Language Processing](https://arxiv.org/abs/2012.03755).
- Grammar-aware QNLP experiments encode word meanings as parameterized quantum circuits and grammar as entangling structure. Source: [Grammar-aware sentence classification on quantum computers](https://arxiv.org/abs/2012.03756).
- Quantum self-attention work translates self-attention into quantum-neural machinery for text classification, but still starts from a classical NLP framing. Source: [Quantum Self-Attention Neural Networks for Text Classification](https://arxiv.org/abs/2205.05625).
- Quantum transformer variants show several ways to replace or approximate transformer attention with quantum circuits, especially in small-data or vision settings. Source: [Quantum Vision Transformers](https://arxiv.org/abs/2209.08167).
- Complex-valued transformer work is a useful near neighbor because it investigates sequence modeling with complex states without requiring quantum hardware. Source: [Complex Transformer](https://arxiv.org/abs/1910.10202).
- Recent phase-coherent transformer work argues that standard softmax token competition can be a poor fit for phase-preserving computation. Source: [Complex-Valued Phase-Coherent Transformer](https://arxiv.org/abs/2605.10123).

## Encoding Directions

### 1. Amplitude/Phase Token States

Each lexical item maps to a normalized complex vector:

```text
|token> = normalize(sum_i amplitude_i * exp(j * phase_i) |basis_i>)
```

Amplitudes answer "which basis features are present?" Phases answer "how should this token interfere with others?"

The earliest prototype should keep amplitudes interpretable:

- character n-grams,
- morphological affixes,
- shape/case features,
- coarse lexical categories when available,
- position or role features as unitary phase shifts, not vector addition.

### 2. Relative Phase as Relation Carrier

Relationships can be modeled as diagonal unitary operators:

```text
U_relation = diag(exp(j * theta_1), ..., exp(j * theta_n))
compatibility(a, relation, b) = |<a| U_relation |b>|^2
```

This gives a crisp toy target: learn or define a phase operator where modifier-noun, subject-verb, or object-verb pairs interfere constructively, while mismatched pairs interfere destructively.

### 3. Structured Tokenizer Instead of BPE

BPE optimizes compression and frequency reuse. It does not care whether a subword has stable Hilbert-space meaning. Quantum-native tokenization should be feature-first:

- a token is a superposition over interpretable feature basis states,
- words sharing morphology naturally share amplitude support,
- relation- or grammar-sensitive features affect phase,
- unknown words degrade gracefully through sub-token features.

This is closer to "state preparation" than token lookup.

### 4. Unitary Context Blocks

A transformer-like block can be reframed as:

1. prepare per-token states,
2. apply positional phase operators,
3. apply relation phase operators,
4. mix neighboring or selected token states with norm-preserving Givens rotations,
5. measure selected observables for logits or losses.

The non-unitary parts should be explicit: measurement, normalization during state preparation, and loss calculation.

## Open Risks

- **Expressivity vs. unitarity:** pure unitary evolution is reversible and norm-preserving; language modeling needs irreversible compression for prediction. The architecture needs explicit measurement or ancilla/discard steps.
- **Scaling:** full sequence Hilbert spaces grow exponentially if modeled literally. The practical route is likely tensor-factorized states, low-rank approximations, or local unitary blocks.
- **Training:** learned phases can collapse to arbitrary encodings unless losses reward interference patterns directly.
- **Tokenizer ambiguity:** feature tokenizers can smuggle classical NLP assumptions into the state basis. That may be acceptable, but it should be measured.

## Minimal Experiments

1. **Toy relation probe:** hand-code feature states and relation phase operators; verify Born overlap separates compatible and incompatible pairs.
2. **Learned relation probe:** fit only phase angles for a synthetic grammar dataset.
3. **Tokenizer comparison:** compare BPE-like IDs, character n-gram real vectors, and complex feature states on OOV generalization.
4. **Unitary block ablation:** compare unconstrained complex linear layers with explicitly unitary diagonal + Givens layers.
5. **Measurement study:** test whether logits from Born probabilities behave differently from logits from real projections of complex states.

## Working Definition

For this project, "quantum-native" means:

- states are complex and normalized,
- phase is semantically meaningful and trainable,
- major contextual transformations are unitary,
- relation scoring uses inner products or Born-rule probabilities,
- tokenization prepares structured states rather than arbitrary IDs.

Hardware execution is optional at this stage.
