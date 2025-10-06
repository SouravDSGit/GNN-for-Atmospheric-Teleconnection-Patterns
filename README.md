# Graph Neural Networks for Atmospheric Teleconnections

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SouravDSGit/GNN-for-Atmospheric-Teleconnection-Patterns/blob/main/GNN_for_Atmospheric_Teleconnection_Patterns.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.x-blue.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A cutting-edge educational notebook** demonstrating how Graph Neural Networks can learn atmospheric teleconnection patterns—the long-distance climate connections that drive extreme weather events worldwide.

## 📚 What You'll Learn

This notebook teaches you how to:

- ✅ Represent the atmosphere as a graph network (nodes = locations, edges = connections)
- ✅ Build Graph Neural Networks (GNNs) for spatial climate data
- ✅ Learn ENSO teleconnection pathways automatically from data
- ✅ Predict extreme events weeks-to-months ahead (subseasonal timescales)
- ✅ Interpret GNN predictions through node importance analysis
- ✅ Apply cutting-edge spatial AI to atmospheric science

**Perfect for:** Climate scientists learning GNNs, AI researchers interested in Earth science, students studying spatial deep learning, anyone curious about teleconnections and extreme weather prediction.

## 🌍 Background: Atmospheric Teleconnections

### What are Teleconnections?

**Teleconnections** are large-scale climate patterns that connect weather across vast distances. When something happens in one part of the world, it affects weather thousands of miles away.

**Famous examples:**

🌊 **El Niño-Southern Oscillation (ENSO)**
- Warm water in tropical Pacific Ocean
- Affects weather globally: droughts in Australia, floods in California, warm winters in Canada
- Operates on seasonal-to-interannual timescales (months to years)
- Most important climate pattern for subseasonal prediction

🌀 **North Atlantic Oscillation (NAO)**
- Pressure difference between Iceland and Azores
- Controls European and East Coast winter weather
- Influences storm tracks and temperatures

🏔️ **Pacific-North American Pattern (PNA)**
- Links tropical Pacific to North American weather
- Major driver of temperature/precipitation extremes
- Critical for weekly-to-monthly forecasts

### Why Teleconnections Matter

**For extreme events:**
- 🔥 2015-16 El Niño → Droughts in Southeast Asia, floods in South America
- ❄️ Strong NAO+ → Mild European winters, cold eastern North America
- 🌡️ 1997-98 El Niño → Record-breaking global temperatures

**For prediction:**
- Teleconnections provide **predictability** at subseasonal-to-seasonal timescales (weeks to months)
- Traditional weather forecasts: 7-10 days
- Climate projections: decades to centuries
- **The gap**: Subseasonal prediction (2 weeks to 2 months) where teleconnections matter most!

### The Problem Traditional Methods Face

**Linear correlation approaches:**
- Assume simple relationships between locations
- Miss non-linear dynamics
- Can't capture multi-step propagation (A→B→C)
- Require manual feature engineering

**What we need:**
- Model the **network structure** of the atmosphere
- Learn **non-linear relationships** automatically
- Capture **multi-hop connections** (how signals propagate)
- Interpretable predictions (which locations matter?)

### Enter Graph Neural Networks

**GNNs are perfect for teleconnections because:**
- ✅ Represent atmosphere as a **network** (graph)
- ✅ Learn **spatial patterns** automatically
- ✅ Capture **long-range connections** (not just nearby locations)
- ✅ Model **information propagation** (how ENSO signal travels)
- ✅ Provide **interpretable results** (node importance = influence)

## 🚀 Getting Started

### Option 1: Google Colab (Recommended - No Setup!)

1. Click the "Open in Colab" badge above
2. Click **Runtime → Run All**
3. Wait ~10-15 minutes for execution
4. Explore the results and visualizations!

**Zero installation required!** PyTorch and PyTorch Geometric install automatically.

### Option 2: Local Jupyter Notebook

```bash
# Clone repository
git clone https://github.com/SouravDSGit/GNN_for_Atmospheric_Teleconnection_Patterns.git
cd GNN-for-Atmospheric-Teleconnection-Patterns

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook GNN_for_Atmospheric_Teleconnection_Patterns.ipynb
```

**Requirements:**
- Python 3.8+
- PyTorch 2.x
- PyTorch Geometric 2.x
- Standard scientific Python libraries

## 📖 What This Notebook Does

### Complete Workflow (13 Sections)

#### **Section 1-2: Setup & Imports**
- Auto-installs PyTorch and PyTorch Geometric
- Imports all required libraries (NetworkX for visualization, etc.)
- Sets up environment for graph-based learning

#### **Section 3: Generate Global Climate Data with Teleconnections**
- Creates 50 years of monthly climate data (600 time steps)
- Simulates 20 global locations as graph nodes
- Generates realistic ENSO cycles (3-7 year periodicity)
- Includes ENSO teleconnection effects with time lags

**Locations include:**
- Niño 3.4 region (tropical Pacific) - the ENSO source
- North Pacific, North America (target region)
- Indo-Pacific, Australia (inverse ENSO response)
- North Atlantic, Europe, Asia
- Global coverage for realistic teleconnections

**Physical realism:**
- ENSO amplitude: ±2°C (matches observations)
- Time lags: 1-2 months (realistic propagation)
- Seasonal modulation: Stronger in boreal winter
- Persistence: Multi-month ENSO episodes

#### **Section 4: Build Atmospheric Graph Network**
- Constructs graph with 20 nodes (locations)
- Creates edges based on:
  - Geographic distance
  - Known teleconnection pathways (ENSO → North Pacific → North America)
  - Historical correlation strength
- Assigns edge weights (connection strength)

**Graph properties:**
- Nodes: 20 global locations
- Edges: ~150-200 connections (varies by distance threshold)
- Directed graph: Captures flow of atmospheric influence
- Weighted edges: Stronger teleconnections have higher weights

**Visualizations:**
- Geographic network map showing node positions
- ENSO time series with El Niño/La Niña events marked
- Clear identification of teleconnection pathways

#### **Section 5: Prepare Graph Data for GNN**
- Creates graph snapshots for each time step
- Node features: temperature, precipitation, SLP, plus lagged values
- Includes temporal encoding (seasonal cycle)
- Generates 597 graph objects for training

**Each graph snapshot contains:**
- 20 nodes with 8 features each
- Current + 3 previous months (temporal context)
- Seasonal encoding (sin/cos of month)
- Labels: ENSO index, extreme heat, extreme precip

#### **Section 6: Define GNN Architecture**
- Builds 3-layer Graph Convolutional Network (GCN)
- Uses batch normalization and dropout
- Multi-task outputs: ENSO regression + extreme classification
- Total parameters: ~50,000 (efficient!)

**Architecture design:**
```
Input: Node features (8 per node)
  ↓
GCN Layer 1 (64 neurons) + ReLU + BatchNorm + Dropout
  ↓
GCN Layer 2 (64 neurons) + ReLU + BatchNorm + Dropout
  ↓
GCN Layer 3 (32 neurons) + ReLU + BatchNorm
  ↓
Global Mean Pooling (aggregate all nodes)
  ↓
├─→ ENSO prediction (regression)
├─→ Extreme heat probability (classification)
└─→ Extreme precip probability (classification)
```

**Why this design?**
- 3 layers = capture multi-hop teleconnections
- Dropout = prevent overfitting
- Global pooling = capture system-wide state
- Multi-task = learn shared atmospheric dynamics

#### **Section 7: Train the GNN**
- Trains for up to 50 epochs with early stopping
- Uses Adam optimizer with learning rate scheduling
- Monitors validation performance
- Saves best model automatically

**Training features:**
- Combined loss (regression + classification)
- Early stopping (patience = 10 epochs)
- Learning rate reduction on plateau
- Training typically completes in 20-30 epochs

#### **Section 8-9: Evaluate Performance & Visualize**
- Tests on held-out data (15% of dataset)
- Calculates comprehensive metrics
- Creates 9-panel result visualization
- Shows ROC curves, confusion matrices, time series

**Evaluation metrics:**
- ENSO prediction: R², MAE, RMSE
- Extreme events: ROC-AUC, precision, recall, F1
- Time series comparison: predicted vs actual

**You'll get:**
- `gnn_teleconnection_results.png` - Comprehensive 9-panel figure
- Classification reports for both extreme types
- ENSO vs extremes relationship analysis

#### **Section 10-11: Node Importance Analysis**
- Uses gradient-based attribution
- Calculates which locations matter most
- Reveals learned teleconnection pathways
- Creates geographic importance maps

**Key questions answered:**
- Which locations drive extreme heat predictions?
- Which locations drive extreme precip predictions?
- Does the GNN identify Niño 3.4 as most important? (Spoiler: Yes!)
- Are the learned pathways physically realistic?

**Outputs:**
- `teleconnection_pathways.png` - Node importance maps
- Rankings of top 10 influential locations
- Physical interpretation of results

#### **Section 12-13: Summary & Complete Analysis**
- Generates comprehensive text report
- Summarizes all findings
- Provides physical interpretation
- Lists implications for operational forecasting

**Output:** `gnn_teleconnection_summary.txt`

### 📊 What Results You'll Get

After running the notebook:

**3 Publication-Quality Figures:**
1. `atmospheric_graph_network.png` - Network structure + ENSO time series
2. `gnn_teleconnection_results.png` - Complete performance analysis
3. `teleconnection_pathways.png` - Learned importance patterns

**1 Summary Report:**
- `gnn_teleconnection_summary.txt` - Full analysis writeup

**Typical Performance:**
- **ENSO Prediction R²**: 0.85-0.90
- **Extreme Heat Detection AUC**: 0.85+
- **Extreme Precip Detection AUC**: 0.85+
- **Most Important Node**: Niño 3.4 region (physically correct!)
- **Teleconnection Pathway**: Niño 3.4 → North Pacific → Western North America (matches known physics!)

## 🔬 Technical Deep Dive

### Why Graphs for Atmospheric Data?

**Traditional approach: Gridded CNNs**
- Treat atmosphere as regular 2D/3D grid
- Convolutional layers capture local patterns
- Good for spatial patterns, images

**Problem with CNNs for teleconnections:**
- Teleconnections are **long-range** (tropical Pacific ↔ North America = 10,000 km!)
- CNNs are **local** (small receptive fields)
- Need many layers to capture global connections
- Computationally expensive for global data

**Graph approach:**
- Nodes = locations (can be far apart)
- Edges = direct connections (even across globe)
- **One GNN layer** can connect distant locations
- Naturally represents atmospheric network structure

### Graph Neural Network Mechanics

**Message Passing (How GNNs Work):**

1. Each node has features (temperature, pressure, etc.)
2. **Aggregate**: Node collects messages from neighbors
3. **Transform**: Apply neural network to messages
4. **Update**: Node updates its representation
5. Repeat for multiple layers

**For teleconnections:**
- Layer 1: Each node learns from direct neighbors
- Layer 2: Nodes learn from 2-hop neighbors (neighbors of neighbors)
- Layer 3: Captures 3-hop connections (e.g., tropics → subtropics → mid-latitudes)

**Why this captures teleconnections:**
- ENSO signal (Node 0: Niño 3.4) propagates through network
- Layer 1: North Pacific receives ENSO information
- Layer 2: North America receives information from North Pacific
- Layer 3: Eastern North America receives information from Western NA
- The GNN learns the **propagation pathway** automatically!

### Edge Construction Strategy

We create edges based on three criteria:

**1. Geographic proximity:**
```python
if distance < threshold:
    create_edge()
```
- Nearby locations influence each other
- Distance threshold: ~50 degrees lat/lon

**2. Known teleconnection patterns:**
```python
teleconnection_pairs = [
    (Niño34, North_Pacific),
    (North_Pacific, Western_NA),
    (Western_NA, Central_NA),
    ...
]
```
- Based on atmospheric science knowledge
- Ensures physically realistic connections

**3. Historical correlation:**
```python
if correlation(temp_A, temp_B) > threshold:
    create_edge()
```
- Data-driven connections
- Captures relationships we might not know about

**Result:** Physics-informed graph structure!

### Multi-Task Learning Benefits

**Why predict ENSO + extremes together?**

1. **Shared representations**: Both tasks learn atmospheric dynamics
2. **Regularization**: Prevents overfitting to one task
3. **Efficiency**: One model instead of three
4. **Physical connection**: ENSO drives extremes, so tasks are related

**Loss function:**
```python
total_loss = loss_enso + 0.5 * loss_heat + 0.5 * loss_precip
```

ENSO prediction guides the model to learn large-scale patterns, which then help predict local extremes.

### Node Importance via Gradients

**How do we know which locations matter?**

We use **gradient-based attribution**:
1. Make a prediction for extreme heat
2. Compute gradient: ∂(prediction)/∂(node_features)
3. Large gradient = node strongly influences prediction

**Physical interpretation:**
- High importance for Niño 3.4 → ENSO is the driver
- High importance for North Pacific → acts as bridge
- High importance for Western NA → target region
- This matches known teleconnection physics!

## 🎓 Educational Value

### For Climate Scientists
- Learn how to apply GNNs to climate data
- Understand graph representations of atmosphere
- See how AI discovers teleconnection pathways
- Bridge physics-based and data-driven approaches

### For AI/ML Researchers
- Apply GNNs to real-world physics problem
- Learn domain-specific graph construction
- See interpretability techniques (gradient attribution)
- Understand multi-task learning benefits

### For Students
- Cutting-edge spatial deep learning
- Complete end-to-end GNN project
- Clear explanations of every step
- Applicable to thesis/research projects

### Compared to Other Approaches

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Linear Correlation** | Simple, interpretable | Only linear, misses propagation | Quick analysis |
| **CNNs** | Good for gridded data | Local, need many layers for global | Images, local patterns |
| **RNNs/LSTMs** | Good for time series | Sequential only, no spatial | Temporal prediction |
| **GNNs (this project!)** | Spatial + non-linear + interpretable | More complex setup | Networked systems, teleconnections |

## 💡 Extension Ideas

Want to take this further?

1. **Full Spatial Resolution**: Use gridded data instead of 20 points
2. **More Teleconnections**: Add NAO, PNA, MJO, AO indices
3. **Temporal GNNs**: Let graph evolve over time (Recurrent GNN)
4. **Attention Mechanisms**: Learn edge weights dynamically
5. **Heterogeneous Graphs**: Different node types (ocean, land, atmosphere)
6. **Real ENSO Data**: Use actual Niño 3.4 SST from NOAA
7. **Seasonal Forecasts**: Predict 3-6 months ahead
8. **Climate Change**: How do teleconnections change under warming?
9. **Ensemble GNNs**: Multiple models for uncertainty
10. **Hybrid Physics-ML**: Combine GNN with numerical model output

## 🤝 Contributing

This is a community resource!

- 🐛 Found a bug? → Open an issue
- 💡 Have an idea? → Submit a pull request
- 📚 Improve docs? → Very welcome!
- 🎓 Using for teaching? → Let us know!

## 📄 License

MIT License - free for education, research, and commercial use!

## 🙏 Acknowledgments & References

This notebook was created as an educational resource based on:

### Atmospheric Teleconnections
- **ENSO**: Philander (1990), "El Niño, La Niña, and the Southern Oscillation"
- **Teleconnection Dynamics**: Wallace & Gutzler (1981), "Teleconnections in the geopotential height field", *Monthly Weather Review*
- **ENSO Impacts**: Ropelewski & Halpert (1987), "Global and regional scale precipitation patterns", *Monthly Weather Review*
- **NAO**: Hurrell (1995), "Decadal trends in the North Atlantic Oscillation", *Science*

### Graph Neural Networks
- **GNN Foundation**: Scarselli et al. (2009), "The graph neural network model", *IEEE Transactions*
- **Graph Convolutions**: Kipf & Welling (2017), "Semi-supervised classification with graph convolutional networks", *ICLR*
- **Message Passing**: Gilmer et al. (2017), "Neural message passing for quantum chemistry", *ICML*
- **PyTorch Geometric**: Fey & Lenssen (2019), "Fast graph representation learning with PyTorch Geometric"

### GNNs for Climate Science
- **Weather Prediction**: Keisler (2022), "Forecasting global weather with graph neural networks"
- **GraphCast**: Lam et al. (2023), "GraphCast: Learning skillful medium-range global weather forecasting", *Science*
- **Climate Modeling**: Kashinath et al. (2021), "Physics-informed machine learning", *Nature Reviews Physics*
- **Spatial-Temporal GNNs**: Wu et al. (2020), "Connecting the dots: Multivariate time series forecasting with graph neural networks"

### Data & Methods
- **ENSO Indices**: NOAA Climate Prediction Center
- **ERA5 Reanalysis**: Hersbach et al. (2020), "The ERA5 global reanalysis", *QJRMS*
- **Teleconnection Indices**: NOAA Physical Sciences Laboratory
- **Extreme Events**: Perkins & Alexander (2013), "On the measurement of heat waves", *Journal of Climate*

### Software & Tools
- **PyTorch**: Paszke et al. (2019), "PyTorch: An imperative style, high-performance deep learning library"
- **PyTorch Geometric**: Fey & Lenssen (2019), "Fast graph representation learning with PyTorch Geometric"
- **NetworkX**: Hagberg et al. (2008), "Exploring network structure, dynamics, and function using NetworkX"
- **NumPy**: Harris et al. (2020), "Array programming with NumPy", *Nature*
- **Pandas**: McKinney (2010), "Data structures for statistical computing in Python"
- **Matplotlib**: Hunter (2007), "Matplotlib: A 2D graphics environment"

### Inspiration
- Climate informatics workshops
- NOAA Climate Prediction Center
- International Research Institute for Climate and Society (IRI)
- Sub-seasonal to Seasonal (S2S) Prediction Project
- GraphCast and AI weather prediction models

## 📧 Contact

- **Issues**: GitHub Issues for bugs/questions
- **Discussions**: GitHub Discussions for ideas/help
- **Email**: [soumukhcivil@gmail.com]

## 🌟 Citation

If you use this notebook for research or teaching:

```bibtex
@software{GNN-for-Atmospheric-Teleconnection-Patterns,
  author = {Sourav Mukherjee},
  title = {Graph Neural Networks for Atmospheric Teleconnections},
  year = {2025},
  url = {https://github.com/SouravDSGit/GNN-for-Atmospheric-Teleconnection-Patterns}
}
```

---

**Happy Learning! 🌊🌍🤖**

*Note: This project uses synthetic data for educational purposes. For research or operational forecasting, use real atmospheric data (ERA5, MERRA-2) and actual ENSO indices from NOAA.*

---

## 🔗 Related Projects

- [AI Emulator for Atmospheric Blocking](https://github.com/SouravDSGit/AI-Emulator-for-Atmospheric-Blocking-Events) 
- [Project 2: Multi-Model Ensemble Analysis](https://github.com/SouravDSGit/Multi-Model-Ensemble-Analysis-of-Extreme-Event-Attribution) 
