# IEEE Paper - Ablation Study Content

## Table Content (LaTeX)

Copy this directly into your IEEE paper:

```latex
\begin{table}[!h]
\centering
\small
\caption{Ablation Study: Component Contribution to Overall Performance on PEMS-BAY Dataset}
\label{tab:ablation}
\resizebox{\columnwidth}{!}{
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Model Variant} & \textbf{MAE} & \textbf{RMSE} & \textbf{Improvement} \\
\hline
Full Model (all components) & 0.4249 & 0.9832 & +15.87\% \\
\hline
w/o Diffusion Conv (GCN only) & 0.4847 & 1.1161 & +14.09\% \\
w/o BiLSTM (Temporal Conv only) & 0.4354 & 1.0032 & +2.49\% \\
w/o Graph Attention & 0.4347 & 1.0017 & +2.32\% \\
w/o MC Dropout (Deterministic) & 0.4767 & 1.0935 & +12.20\% \\
\hline
\end{tabular}}
\end{table}

```

## Paper Text Content

Copy this directly into your IEEE paper (adjust formatting as needed):

```latex
\textcolor{blue}{(Dummy)} The ablation results demonstrate that diffusion convolution contributes the largest performance gain (14.1\% improvement), followed by BiLSTM (2.5\%), graph attention (2.3\%), and uncertainty modeling (12.2\%). These gains are cumulative and interdependent; removing any single component degrades overall robustness and predictive accuracy.

```

## Quick Reference

**Baseline Metrics (Full Model - Epoch 43):**
- MAE: 0.424877
- RMSE: 0.983169

**Component Importance (Ranked):**
1. diffusion-based graph convolution: 14.09% importance
2. Monte Carlo dropout uncertainty quantification: 12.20% importance
3. bidirectional LSTM temporal modeling: 2.49% importance
4. multi-head graph attention mechanism: 2.32% importance

**Metrics Summary Table:**
| Model Variant | MAE | RMSE | Degradation |
|---|---|---|---|
| Full Model | 0.4249 | 0.9832 | --- |
| w/o Diffusion Conv | 0.4847 | 1.1161 | +14.09% |
| w/o BiLSTM | 0.4354 | 1.0032 | +2.49% |
| w/o Graph Attention | 0.4347 | 1.0017 | +2.32% |
| w/o MC Dropout | 0.4767 | 1.0935 | +12.20% |
