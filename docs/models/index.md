# Models overview

Nine entity-alignment methods, grouped by the kind of signal they exploit. Click any card to
open its dedicated page (idea, architecture diagram, losses, training recipe, results and the
debugging lessons that actually made it reach paper level).

<div class="grid cards" markdown>

-   <span class="ea-pill structural">structural</span> **[NAEA](naea.md)** - IJCAI 2019

    ---

    Neighbourhood-aware GAT over translation-consistent neighbour messages, with hard negatives
    and recomputed bootstrapping.

-   <span class="ea-pill structural">structural</span> **[BootEA](bootea.md)** - IJCAI 2018

    ---

    AlignE (limit-based TransE) + alignment-by-swapping + editable MWGM self-training.

-   <span class="ea-pill structural">structural</span> **[AliNet](alinet.md)** - AAAI 2020

    ---

    Gated multi-hop GNN: 1-hop GCN fused with attentional 2-hop, anchored by a relation-aware loss.

-   <span class="ea-pill structural">structural</span> **[GCN-Align](gcnalign.md)** - EMNLP 2018

    ---

    Functionality-weighted adjacency + a shared 2-layer GCN (structure channel).

-   <span class="ea-pill gnn">relation-aware</span> **[KECG](kecg.md)** - EMNLP 2019

    ---

    Shared diagonal multi-head GAT cross-graph + a knowledge-embedding (TransE) loss, alternated.

-   <span class="ea-pill gnn">relation-aware</span> **[MRAEA](mraea.md)** - WSDM 2020

    ---

    Meta-relation-aware GAT (a relation and its inverse differ) + iterative mutual-NN bootstrap.

-   <span class="ea-pill gnn">relation-aware</span> **[RREA](rrea.md)** - CIKM 2020

    ---

    Relational reflection (Householder) aggregation + turn-based CSLS bootstrap. **Top performer.**

-   <span class="ea-pill side">attributes</span> **[JAPE](jape.md)** - ISWC 2017

    ---

    Merged-seed TransE fused with a TF-IDF attribute channel. The paper that introduced DBP15K.

-   <span class="ea-pill side">entity names</span> **[DGMC](dgmc.md)** - ICLR 2020

    ---

    GloVe entity-name features + sparse top-k neighbourhood consensus. Beats the paper on `fr_en`.

</div>

## At a glance

| Model | Encoder | Alignment signal | Loss | Self-training | Eval |
|-------|---------|------------------|------|:-------------:|:----:|
| NAEA | GAT (neighbour messages) | structure | limit-based margin | recomputed | CSLS |
| BootEA | embedding (AlignE) | structure | limit-based TransE + pull | MWGM | CSLS |
| AliNet | gated multi-hop GNN | structure + relations | margin + TransE anchor | optional | CSLS |
| KECG | diagonal multi-head GAT | structure + relations | triplet + TransE | - | CSLS |
| GCN-Align | shared 2-layer GCN | structure | L1 margin | - | L1 / CSLS |
| JAPE | TransE (merged seeds) | structure + attributes | margin + fused AE | - | CSLS |
| DGMC | RelCNN + consensus | entity names | sparse NLL | consensus | top-k |
| MRAEA | meta-relation GAT | structure + relations | L1 margin | mutual-NN | cosine/CSLS |
| RREA | relational-reflection GAT | structure + relations | L1 margin | CSLS mutual-NN | CSLS |

## Shared building blocks

All models reuse the same engine, so reading one makes the rest easy:

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'14px','fontFamily':'Inter, sans-serif','lineColor':'#7d8590','primaryTextColor':'#e6edf3'}}}%%
flowchart LR
    C["<b>configs/your_model.yaml</b><br/><i>hyper-parameters</i>"]:::cfg
    M["<b>models/your_model.py</b><br/><i>encoder + loss</i>"]:::model
    subgraph ENGINE["shared engine - reused by all nine models"]
        direction LR
        D["<b>data.py</b><br/><i>DBP15K · graphs · negatives</i>"]:::data
        T["<b>trainer.py</b><br/><i>train · eval · bootstrap · log</i>"]:::train
        E["<b>utils/metrics.py</b><br/><i>MRR · Hit@k · CSLS</i>"]:::metric
        D --> T --> E
    end
    D --> M --> T
    C -.->|drives| T
    style ENGINE fill:#0d1117,stroke:#30363d,color:#e6edf3
    classDef cfg    fill:#3a2a05,stroke:#d29922,stroke-width:2px,color:#fde68a;
    classDef data   fill:#0c2d6b,stroke:#58a6ff,stroke-width:2px,color:#dbeafe;
    classDef model  fill:#3b0764,stroke:#a371f7,stroke-width:2px,color:#f3e8ff;
    classDef train  fill:#7c2d54,stroke:#f778ba,stroke-width:2px,color:#ffe4f0;
    classDef metric fill:#14532d,stroke:#3fb950,stroke-width:2px,color:#dcfce7;
```

See the [results page](../results.md) for the full benchmark tables and training curves.
