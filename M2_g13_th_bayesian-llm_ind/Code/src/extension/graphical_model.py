"""
Extended graphical model for BayesAgent (Milestone 2).

Represents the extension described in Report Section 3.2:
  X        - observed question/context
  E_i      - observed evidence items (i = 1..m)
  R_i      - hidden reliability of each evidence item, in [0,1]
  Z_j      - hidden reasoning variables (same role as base vPGM's Z1..Zn)
  Y        - query / final answer

Edges:
  X -> R_i, E_i -> R_i
  X -> Z_j, E_i -> Z_j, R_i -> Z_j
  Z_i -> Z_j (i < j)
  Z_j -> Y

This module has NO dependency on vllm / transformers / torch and needs
no GPU or LLM calls to run. It is meant to make the graphical model in
the report a concrete, testable code object, and to implement the
Dirichlet posterior and sequential update formulas (Sections 4.1 and
5.2 of the report) as plain functions that can be unit-tested against
the hand-worked example in Section 6.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

try:
    import graphviz
    _HAS_GRAPHVIZ = True
except ImportError:
    _HAS_GRAPHVIZ = False


# ---------------------------------------------------------------------
# 1. Graph representation: nodes, edges, factorization
# ---------------------------------------------------------------------

@dataclass
class Node:
    name: str
    kind: str     # "observed" | "hidden" | "query"
    domain: str   # e.g. "text", "continuous[0,1]", "categorical"


@dataclass
class Edge:
    parent: str
    child: str


@dataclass
class GraphicalModel:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, name: str, kind: str, domain: str) -> None:
        self.nodes[name] = Node(name, kind, domain)

    def add_edge(self, parent: str, child: str) -> None:
        assert parent in self.nodes, f"unknown parent node: {parent}"
        assert child in self.nodes, f"unknown child node: {child}"
        self.edges.append(Edge(parent, child))

    def parents_of(self, node_name: str) -> List[str]:
        return [e.parent for e in self.edges if e.child == node_name]

    def children_of(self, node_name: str) -> List[str]:
        return [e.child for e in self.edges if e.parent == node_name]

    def describe(self) -> None:
        print(f"GraphicalModel: {len(self.nodes)} nodes, {len(self.edges)} edges")
        for n in self.nodes.values():
            pa = self.parents_of(n.name)
            print(f"  {n.name:6s} ({n.kind:8s}, {n.domain:16s})  parents={pa}")


def draw(gm: "GraphicalModel", title: str, out_path: str, fmt: str = "png") -> None:
    """
    Render a GraphicalModel as an actual diagram (boxes + arrows) using
    Graphviz. Node styling follows a simple, paper-style convention:
      observed -> plain rectangle, light grey fill
      hidden   -> ellipse, light blue fill
      query    -> ellipse, bold border, light orange fill (the answer Y)
    Writes out_path + "." + fmt, e.g. "extension/base_model.png".
    """
    if not _HAS_GRAPHVIZ:
        raise RuntimeError("graphviz package not installed; run: pip install graphviz")

    dot = graphviz.Digraph(comment=title, format=fmt)
    dot.attr(rankdir="LR", fontsize="12", label=title, labelloc="t")

    style_by_kind = {
        "observed": dict(shape="box", style="filled", fillcolor="#e0e0e0"),
        "hidden":   dict(shape="ellipse", style="filled", fillcolor="#bbdefb"),
        "query":    dict(shape="ellipse", style="filled", fillcolor="#ffe0b2",
                          penwidth="2"),
    }

    for n in gm.nodes.values():
        dot.node(n.name, n.name, **style_by_kind.get(n.kind, {}))

    for e in gm.edges:
        dot.edge(e.parent, e.child)

    rendered_path = dot.render(filename=out_path, cleanup=True)
    print(f"Saved diagram to {rendered_path}")


def build_base_vpgm_model_actual_n3() -> GraphicalModel:
    """
    The EXACT structure used by the official repo for vpgm-n3 on
    ScienceQA, taken verbatim from prompts/scienceqa_new.py:

        "BN structure: Q,C,M,I,K->Z1; I->Z2; I->Z3; and Q,Z1,Z2,Z3->Y"

    Observed variables, per the prompt, are Q, C, M, I, K (not a
    single merged X) -- Z1 depends on all five, Z2 and Z3 depend only
    on I, and Z1/Z2/Z3 do NOT depend on each other. This replaces the
    earlier placeholder build_base_vpgm_model(), which incorrectly
    assumed a Z1->Z2->Z3 chain.
    """
    gm = GraphicalModel()
    for obs in ["Q", "C", "M", "I", "K"]:
        gm.add_node(obs, "observed", "text")

    gm.add_node("Z1", "hidden", "binary (0/1)")
    for obs in ["Q", "C", "M", "I", "K"]:
        gm.add_edge(obs, "Z1")

    gm.add_node("Z2", "hidden", "binary (0/1)")
    gm.add_edge("I", "Z2")

    gm.add_node("Z3", "hidden", "binary (0/1)")
    gm.add_edge("I", "Z3")

    gm.add_node("Y", "query", "categorical")
    gm.add_edge("Q", "Y")
    gm.add_edge("Z1", "Y")
    gm.add_edge("Z2", "Y")
    gm.add_edge("Z3", "Y")

    return gm


def build_base_vpgm_model(n_latent: int = 3) -> GraphicalModel:
    """
    A generic, simplified placeholder (X -> Z1..Zn, Zi -> Zj for i<j,
    Z -> Y). NOT the exact structure used in the official code for
    ScienceQA -- see build_base_vpgm_model_actual_n3() for that, and
    the report/README for why this simplification does not match the
    real prompt. Kept for comparison / as a generic n-latent-variable
    illustration.
    """
    gm = GraphicalModel()
    gm.add_node("X", "observed", "text")
    for j in range(1, n_latent + 1):
        gm.add_node(f"Z{j}", "hidden", "continuous[0,1]")
        gm.add_edge("X", f"Z{j}")
        for k in range(1, j):
            gm.add_edge(f"Z{k}", f"Z{j}")
    gm.add_node("Y", "query", "categorical")
    for j in range(1, n_latent + 1):
        gm.add_edge(f"Z{j}", "Y")
    return gm


def build_extended_model(m_evidence: int = 2, n_latent: int = 3) -> GraphicalModel:
    """
    The extended model from Report Section 3.2: adds E_i, R_i for each
    of m_evidence evidence items, wired into the base vPGM structure.
    m_evidence = 0 should reduce to build_base_vpgm_model(n_latent).
    """
    gm = GraphicalModel()
    gm.add_node("X", "observed", "text")

    for i in range(1, m_evidence + 1):
        gm.add_node(f"E{i}", "observed", "text")
        gm.add_node(f"R{i}", "hidden", "continuous[0,1]")
        gm.add_edge("X", f"R{i}")
        gm.add_edge(f"E{i}", f"R{i}")

    for j in range(1, n_latent + 1):
        gm.add_node(f"Z{j}", "hidden", "continuous[0,1]")
        gm.add_edge("X", f"Z{j}")
        for i in range(1, m_evidence + 1):
            gm.add_edge(f"E{i}", f"Z{j}")
            gm.add_edge(f"R{i}", f"Z{j}")
        for k in range(1, j):
            gm.add_edge(f"Z{k}", f"Z{j}")

    gm.add_node("Y", "query", "categorical")
    for j in range(1, n_latent + 1):
        gm.add_edge(f"Z{j}", "Y")

    return gm


# ---------------------------------------------------------------------
# 2. Factors / formulas (Report Sections 4.1 and 5.2)
# ---------------------------------------------------------------------

def dirichlet_posterior_mean(alpha, n_votes):
    """
    Report Eq. (Section 4.1):
        pi_k = (n_k + alpha_k) / sum_j (n_j + alpha_j)
    alpha, n_votes: array-like of length K.
    Returns the posterior mean pi (array of length K).
    """
    alpha = np.asarray(alpha, dtype=float)
    n_votes = np.asarray(n_votes, dtype=float)
    assert alpha.shape == n_votes.shape
    post = n_votes + alpha
    return post / post.sum()


def alpha_from_prior(lmbda, p):
    """alpha_k = lambda * p_k  (Section 4.1)."""
    return lmbda * np.asarray(p, dtype=float)


def sequential_update(alpha_t, r, kappa, s_bar):
    """
    Report Eq. (Section 5.2):
        alpha_{t+1} = alpha_t + r * kappa * s_bar
    r: reliability in [0,1], kappa: trust constant, s_bar: evidence's
    own answer distribution (array of length K, sums to 1).
    Returns alpha_{t+1} (array of length K).
    """
    alpha_t = np.asarray(alpha_t, dtype=float)
    s_bar = np.asarray(s_bar, dtype=float)
    return alpha_t + r * kappa * s_bar


def belief_from_alpha(alpha):
    """pi = alpha / sum(alpha)."""
    alpha = np.asarray(alpha, dtype=float)
    return alpha / alpha.sum()


def confidence_change(pi_t, pi_t1, y0_index):
    """
    Report Section 5.2: Delta C = C_{t+1} - C_t where C_t = pi_t[y0].
    y0_index: index of the fixed reference answer y0 in pi.
    """
    return pi_t1[y0_index] - pi_t[y0_index]


# ---------------------------------------------------------------------
# 3. Unit test: reproduce the worked example in Report Section 6
# ---------------------------------------------------------------------

def test_worked_example(verbose: bool = True) -> bool:
    """
    K = 3, prior p = (0.6, 0.3, 0.1), lambda = 5
      -> alpha = (3, 1.5, 0.5)
    Votes n = (2, 1, 0) -> posterior alpha = (5, 2.5, 0.5), A = 8
      -> pi_before = (0.625, 0.3125, 0.0625)
    New evidence for class 2: s_bar = (0, 1, 0), r = 0.5, kappa = 1
      -> alpha_after = (5, 3, 0.5), A = 8.5
      -> pi_after = (0.5882, 0.3529, 0.0588)
    """
    p = [0.6, 0.3, 0.1]
    lmbda = 5
    alpha_prior = alpha_from_prior(lmbda, p)
    assert np.allclose(alpha_prior, [3.0, 1.5, 0.5]), alpha_prior

    n_votes = [2, 1, 0]
    pi_before_from_posterior = dirichlet_posterior_mean(alpha_prior, n_votes)
    assert np.allclose(pi_before_from_posterior, [0.625, 0.3125, 0.0625], atol=1e-4), pi_before_from_posterior

    # posterior alpha after votes, before new evidence item
    alpha_before_evidence = alpha_prior + np.array(n_votes)
    assert np.allclose(alpha_before_evidence, [5.0, 2.5, 0.5]), alpha_before_evidence

    # new evidence item arrives: r = 0.5, kappa = 1, s_bar = (0,1,0)
    alpha_after = sequential_update(alpha_before_evidence, r=0.5, kappa=1, s_bar=[0, 1, 0])
    assert np.allclose(alpha_after, [5.0, 3.0, 0.5]), alpha_after

    pi_before = belief_from_alpha(alpha_before_evidence)
    pi_after = belief_from_alpha(alpha_after)

    expected_pi_after = np.array([0.5882, 0.3529, 0.0588])
    ok = np.allclose(pi_after, expected_pi_after, atol=1e-3)

    if verbose:
        print("alpha (prior from lambda*p):      ", alpha_prior)
        print("alpha (after 2,1,0 votes):         ", alpha_before_evidence)
        print("pi before new evidence:            ", np.round(pi_before, 4))
        print("alpha after sequential_update:     ", alpha_after)
        print("pi after new evidence:             ", np.round(pi_after, 4))
        print("expected (from report, by hand):   ", expected_pi_after)
        print("match:", ok)

        # also check C1: r = 0 changes nothing
        alpha_r0 = sequential_update(alpha_before_evidence, r=0.0, kappa=1, s_bar=[0, 1, 0])
        pi_r0 = belief_from_alpha(alpha_r0)
        print("\n(C1 check) pi with r=0 (should equal pi before):", np.round(pi_r0, 4))
        assert np.allclose(pi_r0, pi_before)

    return ok


if __name__ == "__main__":
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== Base vPGM graphical model: generic placeholder (n=3) ===")
    build_base_vpgm_model(n_latent=3).describe()

    print("\n=== Base vPGM graphical model: ACTUAL structure used by official code for vpgm-n3 ===")
    print("(from prompts/scienceqa_new.py: \"Q,C,M,I,K->Z1; I->Z2; I->Z3; and Q,Z1,Z2,Z3->Y\")")
    base_gm = build_base_vpgm_model_actual_n3()
    base_gm.describe()

    print("\n=== Extended graphical model (this project, m=2 evidence, n=3 latent) ===")
    ext_gm = build_extended_model(m_evidence=2, n_latent=3)
    ext_gm.describe()

    if _HAS_GRAPHVIZ:
        print("\n=== Drawing diagrams ===")
        draw(base_gm, "Base vPGM model: actual vpgm-n3 structure (ScienceQA)", os.path.join(out_dir, "base_model"))
        draw(ext_gm, "Extended model (m=2 evidence, n=3 latent)", os.path.join(out_dir, "extended_model"))
    else:
        print("\n(graphviz not installed -- skipping diagram rendering)")

    print("\n=== Worked example check (Report Section 6) ===")
    result = test_worked_example(verbose=True)
    print("\nWORKED EXAMPLE TEST:", "PASSED" if result else "FAILED")
