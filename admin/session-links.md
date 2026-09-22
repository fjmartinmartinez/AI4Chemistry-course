# Per-session source links (v0.1, 2026-09-02)

Companion to `syllabus.md`. "Core" = material the session notebook/slides are built from; "Alt/Colab" = ready-to-run alternatives for students. All links verified 2026-09-02.

## Week 01 — Python bootcamp
- A core: MolSSI 01 Introduction https://education.molssi.org/python_scripting_cms/01-introduction/index.html ; 02 File parsing https://education.molssi.org/python_scripting_cms/02-file_parsing/index.html ; 06 Functions https://education.molssi.org/python_scripting_cms/06-functions/index.html ; SciComp ch.0–1 https://weisscharlesj.github.io/SciCompforChemists/notebooks/chapter_00/chap_00_notebook.html , https://weisscharlesj.github.io/SciCompforChemists/notebooks/chapter_01/chap_01_notebook.html
- A alt (Colab): EPFL 01a Python crash course https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01a_python_crash_course.ipynb
- B core: SciComp ch.4 NumPy https://weisscharlesj.github.io/SciCompforChemists/notebooks/chapter_04/chap_04_notebook.html ; ch.3 Matplotlib https://weisscharlesj.github.io/SciCompforChemists/notebooks/chapter_03/chap_03_notebook.html ; MolSSI 04 Tabular data https://education.molssi.org/python_scripting_cms/04-tabular_data/index.html ; 05 Plotting https://education.molssi.org/python_scripting_cms/05-plotting/index.html

## Week 02 — Data + molecules in code
- A core: SciComp ch.5 Pandas https://weisscharlesj.github.io/SciCompforChemists/notebooks/chapter_05/chap_05_notebook.html ; EPFL 01b Pandas https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01b_python_essentials_pandas.ipynb ; EPFL 01d RDKit basics https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01d_rdkit_basics.ipynb
- B core: TeachOpenCADD T001 ChEMBL https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html ; T002 ADME filtering https://projects.volkamerlab.org/teachopencadd/talktorials/T002_compound_adme.html ; T005 clustering https://projects.volkamerlab.org/teachopencadd/talktorials/T005_compound_clustering.html ; EPFL 01c plotting https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/01%20-%20Basics/01c_python_essentials_plotting.ipynb

## Week 03 — Regression & model assessment
- A core: dmol Intro to ML https://dmol.pub/ml/introduction.html ; dmol Regression & model assessment https://dmol.pub/ml/regression.html
- B core: Walters regression model https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/regression_model.ipynb ; Walters cross-validation https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/cross_validation.ipynb ; gentler alt: ML4chemArg repo https://github.com/ML4chemArg/Intro-to-Machine-Learning-in-Chemistry

## Week 04 — Classification & unsupervised
- A core: dmol Classification https://dmol.pub/ml/classification.html ; dmol Kernel learning (qualitative) https://dmol.pub/ml/kernel.html
- B core: Walters classification https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/classification_model.ipynb ; comparing classifiers https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/comparing_classification_models.ipynb ; TeachOpenCADD T007 ML screening https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html ; alt: ML-in-chemistry-101 https://github.com/BingqingCheng/ML-in-chemistry-101

## Week 05 — Deep learning I
- A core: dmol Tensors & shapes https://dmol.pub/math/tensors-and-shapes.html ; DL overview https://dmol.pub/dl/introduction.html ; Standard layers https://dmol.pub/dl/layers.html
- B core: EPFL intro to DL https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/01_intro_to_dl.ipynb

## Week 06 — Representations & inductive bias
- A core: dmol Input data & equivariances https://dmol.pub/dl/data.html ; survey only: Equivariant NNs https://dmol.pub/dl/Equivariant.html
- B core: DeepChem MoleculeNet intro https://github.com/deepchem/deepchem/blob/master/examples/tutorials/An_Introduction_To_MoleculeNet.ipynb (representation shoot-out data source)

## Week 07 — Graph neural networks
- A core: dmol GNNs https://dmol.pub/dl/gnn.html
- B core: EPFL graph NNs https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/02_graph_nns.ipynb ; EPFL GNN/Chemprop example https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/03%20-%20Intro%20to%20Deep%20Learning/03_gnn_simple_example.ipynb ; alt: DeepChem graph convolutions https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Introduction_to_Graph_Convolutions.ipynb

## Week 08 — Sequences, attention, chemical LLMs
- A core: dmol Attention https://dmol.pub/dl/attention.html ; DL on sequences https://dmol.pub/dl/NLP.html ; survey: Pretraining https://dmol.pub/dl/pretraining.html
- B core: EPFL reaction prediction (template-free) https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/07%20-%20Reaction%20Prediction/template_free.ipynb

## Week 09 — Generative models & XAI
- A core: dmol VAE https://dmol.pub/dl/VAE.html ; Explaining predictions https://dmol.pub/dl/xai.html ; survey slide: Flows https://dmol.pub/dl/flows.html
- B core: EPFL molecular generative models https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/05%20-%20Generative%20Models/Molecular%20Generative%20Models.ipynb ; EPFL SMILES-LSTM walkthrough https://colab.research.google.com/github/schwallergroup/ai4chem_course/blob/main/notebooks/06%20-%20Generative%20Models%202/SMILES-LSTM-Walkthrough.ipynb ; in-browser demo: dmol generative RNN https://dmol.pub/applied/MolGenerator.html

## Week 10 — Applications, pitfalls, project
- A core: dmol Predicting DFT energies with GNNs https://dmol.pub/applied/QM9.html ; Modern molecular NNs https://dmol.pub/dl/molnets.html ; pitfalls lab material: Walters comparing regression models https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/comparing_regression_models.ipynb
- B: mini-project hackathon — no new material; datasets via MoleculeNet (wk-06 link) and course `sources/datasets/`.
