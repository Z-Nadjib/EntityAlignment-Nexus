"""Models for entity alignment: NAEA, BootEA, AliNet, KECG, GCN-Align, JAPE, DGMC, MRAEA and RREA."""
from .naea import NAEA, alignment_loss, margin_ranking_loss
from .bootea import BootEA, alignment_pull_loss, limit_based_triple_loss
from .alinet import AliNet, alinet_align_loss
from .kecg import KECG, kecg_cg_loss, kecg_ke_loss
from .gcnalign import GCNAlign, gcnalign_loss
from .jape import JAPE, jape_se_loss
from .dgmc import DGMC, RelCNN, RelConv
from .mraea import MRAEA, mraea_align_loss
from .rrea import RREA

__all__ = [
    "NAEA", "alignment_loss", "margin_ranking_loss",
    "BootEA", "alignment_pull_loss", "limit_based_triple_loss",
    "AliNet", "alinet_align_loss",
    "KECG", "kecg_cg_loss", "kecg_ke_loss",
    "GCNAlign", "gcnalign_loss",
    "JAPE", "jape_se_loss",
    "DGMC", "RelCNN", "RelConv",
    "MRAEA", "mraea_align_loss",
    "RREA",
]
