# SkylineAGI-Mv4
Skyline Artificial General intelligence. AGI Machine Learning Model

Original Start date: 11/23/2022
Project Creation Location: 
Philippines, Mindanao, Davao Del Sur, Santacruz

## **README for SkylineAGI-Mv4**

### **Project Overview**
SkylineAGI-Mv4 is an Artificial General Intelligence (AGI) framework focusing on dynamic problem-solving and learning without relying on massive datasets like traditional Large Language Models (LLMs). It uses a **multi-tiered database structure** to manage domain-specific knowledge efficiently.

### **Key Features**
- **Dynamic Knowledge Base**: Centralized access to domain datasets, organized into 9 levels with 3 complexity tiers per level.
- **Flexible Architecture**: Designed for adaptability with modular components for assimilation, self-learning, and dynamic hyperparameter tuning.
- **Efficiency-Driven**: Operates on targeted datasets to reduce computational overhead, focusing on relevance rather than volume.
- **Dynamic Adaptation**: Continuously adjusts based on domain complexity and evolving objectives.

### **Folder and File Structure**
1. **`src/`**: Core source code, including logic for dynamic data handling and self-learning.
2. **`data/`**: Stores domain-specific datasets (e.g., `math_dataset1.csv` to `math_dataset9.csv`).
3. **`config.json`**: Central configuration file for dynamic activation functions, complexity factors, and knowledge base settings.
4. **`domain_dataset.json`**: Master dataset list referencing file paths and metadata.
5. **`docs/`**: Documentation for system design and API details.
6. **`tests/`**: Unit tests to validate system functionality.

### **Requirements**
1. Current running environment:
   ```bash
   Test Environment: linux riplit
   Python Tools 3.12.5
   pyright-extended 2.0.12
   ```
### **Computers Used**
1: Cellphone = Infinix Note12.

### **Environments**
Environment Tools
```bash
  Remote Test Environments: 
1: Replit
2: Google Colab

   Free Construction AI: 
0: POE
1: Claude 3 Haiku-200k, 
2: Perplexity, -GitHub capable
3: Chatgpt, -GitHub capable
4: Cortex, Semi GitHub capable
5: Bard/Palm/Gemini.

   Free Code Checking AI:
0: POE 
1: Blackbox.ai, CyberCoder
2: Chatgpt, 
3: Perplexity, 
4: Google Gemini, 
5: Nvidia Llama 3.2 Nemotron
6: Claude 3 Haiku-200k
7: Cortex
8: Google Colab Compiler
9: Replit Compiler -GitHub capable
   ```

### **Dependencies**
Environment Dependencies:
   ```bash
   Python Tools 3.12.5
   pyright-extended 2.0.12
   ```

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/rainmanp7/SkylineAGI-Mv4.git
   cd SkylineAGI-Mv4
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the knowledge base in `config.json`.

### **Usage**
1. Add datasets to the `data/` folder.
2. Define domain-specific complexity in `domain_dataset.json`.
3. Run the main AGI script:
   ```bash
   python main.py

  ```

#### **Dynamic Data Loading**
- Enhance dynamic loading with preprocessing based on complexity:
  ```python
  def preprocess_data(dataset, complexity):
      # Add preprocessing logic for each complexity tier
      pass
  ```

---

### **Next Steps**
Update the `tests/` folder to validate dataset integration.


## About
SkylineAGI-Mv4 is an Artificial General Intelligence (AGI) framework designed to serve as a cognitive "brain" for Artificial Intelligence (AI) systems. Unlike traditional Large Language Models (LLMs), this AGI framework is focused on dynamic problem-solving, cross-domain knowledge integration, and efficient learning without relying on massive datasets.

The key innovation of SkylineAGI-Mv4 is its multi-tiered knowledge base structure. Each domain (e.g., Mathematics, Medicine, Military Strategy) is represented by a hierarchical set of 9 datasets, each with 3 distinct complexity tiers. This allows the AGI to efficiently manage and access domain-specific knowledge, with a total of 27 complexity realms per domain.

The knowledge graph is structured in a way that enables consistent and scalable integration of information. As the AGI system ingests and processes data, it associates each piece of information with the appropriate domain and complexity level, creating a rich and interconnected knowledge base.

The assimilation process in SkylineAGI-Mv4 is designed to seamlessly integrate new information into the existing knowledge base. When new data is encountered, the system analyzes its complexity and domain relevance, then assimilates it into the appropriate tier and level of the knowledge graph. This dynamic assimilation allows the AGI to continuously expand and refine its understanding, adapting to evolving information and requirements.

This design allows the SkylineAGI-Mv4 to be accessed via a REST API, enabling seamless integration with a wide range of Artificial Intelligence systems. Multiple instances of the AGI can be deployed, and these instances can collaborate and share knowledge to provide more comprehensive and insightful responses.

By leveraging this structured knowledge base, the efficient assimilation process, and the ability to handle varying levels of complexity, the SkylineAGI-Mv4 framework aims to serve as a versatile and adaptive "brain" for AI systems, enabling them to engage in dynamic problem-solving, cross-domain generalization, and efficient learning.

The potential applications of SkylineAGI-Mv4 are vast and span a wide range of industries:

- **Medical**: Assisting in medical diagnosis, drug discovery, and personalized treatment planning by integrating cross-disciplinary knowledge from domains like biology, pharmacology, and clinical studies.
- **Military**: Enhancing strategic decision-making, logistics optimization, and intelligence analysis by leveraging the AGI's ability to process and synthesize information from diverse sources, such as geopolitics, military tactics, and technological developments.
- **Finance**: Improving financial modeling, risk assessment, and investment strategies by drawing insights from economic trends, market data, and regulatory frameworks.
- **Sustainability**: Supporting the development of sustainable solutions by combining knowledge from fields like environmental science, energy systems, and urban planning.

By enabling AI systems to learn from each other and share their collective knowledge, SkylineAGI-Mv4 can accelerate the advancement of Artificial Intelligence and empower it to tackle complex, real-world challenges more effectively. This framework represents a significant step towards the development of truly intelligent and collaborative AI systems.
