# Title
Convolutions are a simple yet powerful technique to maximize computing efficiency when it comes to many types of data. For graphs however, this technique is not so easy to apply as it is quite hard to determine where exactly we are in a graph, where we are moving and what neighbors to include in the convolution; which  are all important information we need to calculate the convolution. Therefore, previous methods used sepctral theory, eigenvalues, eigenvectors and fourier transforms to apply the convolution in a space where we know where we are at the price of high compute of $O(N^2)$ to $O(N^3)$.

### Context / Key-concepts
Kipf and Welling simplified this spectral formulation using a first-order approximation of Chebyshev polynomials, reducing the complexity to $O(|\mathcal{E}|)$. Combined with the renormalization trick, this yields the well-known propagation rule:

$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^lW^l) = \sigma(\hat{A}H^lW^l)$,
 
where $\tilde{A} = A + I$ adds self-loops, $\tilde{D}$ is the corresponding degree matrix, and $\hat{A}$ denotes the normalized adjacency matrix. The normalization prevents feature magnitudes from growing uncontrollably and reduces the dominance of high-degree nodes during aggregation. Self-loops are important because they allow each node to preserve and propagate its own features in addition to information received from its neighbors.

## Implementation details

The normalization prevents feature magnitudes from growing uncontrollably and reduces the dominance of high-degree nodes during aggregation. Self-loops are important because they allow each node to preserve and propagate its own features in addition to information received from its neighbors. Each layer therefore follows the above mentioned propagation rule, with ReLU activation after the first layer and dropout applied between the two layers.

In addition to the full-batch approach of Kipf and Welling, a mini-batch variant was implemented to compare runtime and scalability on larger datasets. While the original model is designed for full-batch training on citation graphs such as Cora, the mini-batch version was added to investigate how the training behavior changes once the graph becomes too large for an efficient full-graph forward pass. 

While the original Kipf-Welling model is designed for full-batch training on citation graphs such as Cora, the mini-batch version was added to investigate how the training behavior changes once the graph becomes too large for an efficient full-graph forward pass.


## Results

To ensure comparability for all runs the same standard NVIDIA L4 GPU with 24 GB GDDR6 was used. 

### Result full-batch vs. Kipf-welling

| Setting         | Model   | text       | text (%)       |
|-----------------|---------|------------|----------------|
| text            | text    | 100        | -              |
|                 | text    | 100        | 19.28%         |
| text            | text    | 100        | -              |
|                 | text    | 100        | 11.12%         |

text

plots

### Result mini-batch vs. full-batch on different data sets

| Setting         | Model   | text       | text (%)       |
|-----------------|---------|------------|----------------|
| text            | text    | 100        | -              |
|                 | text    | 100        | 19.28%         |
| text            | text    | 100        | -              |
|                 | text    | 100        | 11.12%         |


### Findings

text

plots


## Project structure

```
repo-name/
├── README.md
├── requirements.txt                 
├── src/
│   ├── model.py                   
│   ├── train.py                
│   ├── data_utils.py              
├── data/
│   └── Planetoid
│   └── ogbn_arxiv                
├── notebooks/
│   └── experiments.ipynb  
├── configs/
│   └── default_config.yaml
│   └── citeseer.yaml
│   └── pubmed.yaml   
│   └── ogbn_arxiv_full_batch.yaml
│   └── pubmed_config_mini_batch.yaml
├── results/
    └── ... 
```

## How to run

Main packages used in this project:
- PyTorch
- PyTorch Geometric
- Pyg-lib
- NumPy
- SciPy
- PyYAML

Pyg-lib was used to obtain the subgraphs for the mini-batch approach. However, the pyg-lib package is only compatible with no later version of torch then 2.8.0 and cu128. Install the ground packages via: 

`pip install -r requirements.txt`

and manually install pyg-lib (assumung torch 2.8.0 and cuda cu128) with:

`pip install pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html`

Run default experiment:

`python src/train.py --config configs/default_config.yaml`

To run different experiments I used different config.yaml files, as this is more straight forward and clearer then using many different flags for every run.

| Flag              | Type    | Default | Description |
|------------------|---------|---------|-------------|
| `--config`        | text     | text    | text |


## References

text