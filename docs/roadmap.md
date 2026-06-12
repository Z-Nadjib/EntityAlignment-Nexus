# Roadmap

EntityAlignment-Nexus is built to **grow into the reference hub for entity-alignment models**. The current
release covers nine classic structural, relation-aware, attribute- and name-based methods. The
next wave is **transformer-based** EA.

## Vision

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'14px','fontFamily':'Inter, sans-serif','lineColor':'#7d8590','primaryTextColor':'#e6edf3'}}}%%
flowchart LR
    subgraph PH1["Phase 1 - shipped (9 models)"]
        direction TB
        P1["NAEA · BootEA · AliNet · GCN-Align<br/>KECG · JAPE · DGMC · MRAEA · RREA"]:::done
    end
    subgraph PH2["Phase 2 - next"]
        direction TB
        P2["Transformer encoders<br/>PLM-initialised EA<br/>Dangling-aware EA"]:::next
    end
    subgraph PH3["Phase 3 - later"]
        direction TB
        P3["Multi-modal EA<br/>LLM-assisted EA<br/>Unsupervised / zero-seed EA"]:::later
    end
    PH1 ==> PH2 ==> PH3
    style PH1 fill:#0d1117,stroke:#3fb950,color:#e6edf3
    style PH2 fill:#0d1117,stroke:#58a6ff,color:#e6edf3
    style PH3 fill:#0d1117,stroke:#a371f7,color:#e6edf3
    classDef done  fill:#14532d,stroke:#3fb950,stroke-width:2px,color:#dcfce7;
    classDef next  fill:#0c2d6b,stroke:#58a6ff,stroke-width:2px,color:#dbeafe;
    classDef later fill:#3b0764,stroke:#a371f7,stroke-width:2px,color:#f3e8ff;
```

## Planned: transformer-based EA

<div class="grid cards" markdown>

-   :material-transit-connection-variant: **Self-attention encoders**

    ---

    Graph-transformer aggregators that replace fixed-hop GAT with global attention over the
    neighbourhood (e.g. relation-aware transformer layers on the KG).

-   :material-text-box-multiple: **PLM-initialised alignment**

    ---

    Initialise entity features with pre-trained multilingual language models (mBERT, XLM-R,
    LaBSE) instead of GloVe, in the spirit of BERT-INT / SelfKG.

-   :material-link-off: **Dangling-aware EA**

    ---

    Handle entities with **no** counterpart (the DBP2.0 / dangling setting), a more realistic
    open-world variant of the task.

-   :material-robot-happy: **LLM-assisted EA**

    ---

    Use large language models as candidate re-rankers or verifiers on top of a cheap structural
    retriever.

</div>

## Candidate models (shortlist)

| Model | Venue | Why it fits |
|-------|:-----:|-------------|
| **BERT-INT** | IJCAI 2020 | BERT-based interaction model over names/descriptions/attributes |
| **SelfKG** | WWW 2022 | self-supervised, (almost) no seed alignments |
| **Dual-AMN** | WWW 2021 | proxy-attention, very fast and strong on DBP15K |
| **TransEdge** | ISWC 2019 | edge-centric translational embeddings |
| **EVA / MMEA** | AAAI 2021 | multi-modal (images + structure + attributes) |

These are **candidates**, not commitments - priority follows community interest and
reproducibility.

## How to propose or contribute a model

1. Open an issue describing the method and its DBP15K numbers.
2. Add `code/src/models/<your_model>.py` (encoder + loss) following the existing pattern.
3. Add a trainer (or reuse one) in `code/src/trainer.py` and a `configs/<your_model>.yaml`.
4. Add a self-contained notebook and a docs page mirroring the others.

The shared `data.py` / `metrics.py` mean you mostly write the model itself. See
[About & contributing](about.md) for the conventions.

!!! question "Want a specific model next?"
    Tell us which transformer-based EA model you want first - community demand drives the order.
