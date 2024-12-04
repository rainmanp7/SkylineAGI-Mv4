# SkylineAGI-Mv4
Skyline Artificial General intelligence. AGI Machine Learning Model

To create a detailed and functional README for your **SkylineAGI-Mv4** repository, we need to align it with the project's structure, goals, and code functionalities as they currently exist. Based on the repository and your provided details, here's a proposed plan for the README content, along with actions to address dataset issues:

---

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

### **Current Issues and Fixes**
#### **Dataset Management**
- **Problem**: Multiple references to datasets in the code create inconsistencies.
- **Fix**: Centralize dataset access using `domain_dataset.json`. Update all code references to route through this file.
  - For example, replace direct file calls like `open('math_dataset1.csv')` with a function:
    ```python
    def get_dataset_path(domain, level):
        with open('domain_dataset.json') as f:
            datasets = json.load(f)
        return datasets[domain][f"level_{level}"]
    ```

#### **Knowledge Base Definition**
- **Problem**: The knowledge base isn't clearly defined for all modules.
- **Fix**: Update `config.json` to point to `domain_dataset.json` for knowledge base handling:
  ```json
  "knowledge_base": "domain_dataset.json"
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
1. Refactor code to ensure all dataset references use `domain_dataset.json`.
2. Review all modules for compliance with centralized dataset access.
3. Update the `tests/` folder to validate dataset integration.
