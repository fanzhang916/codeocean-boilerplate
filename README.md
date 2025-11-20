# Code Ocean Reproducibility Boilerplate

This repository provides a boilerplate structure for creating reproducible computational capsules on Code Ocean, based on best practices for journal submissions. It aims to streamline the process of packaging your research code and data for review and publication, ensuring that your results can be easily verified and reproduced.

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Customizing the Boilerplate](#customizing-the-boilerplate)
  - [Local Development with Docker Compose](#local-development-with-docker-compose)
  - [Running on Code Ocean](#running-on-code-ocean)
- [Example: Neural Network with PyTorch (Tabular Data)](#example-neural-network-with-pytorch-tabular-data)
- [Contributing](#contributing)
- [License](#license)

## Features

-   **Standardised Structure**: Follows a logical and common directory structure for research projects.
-   **Code Ocean Ready**: Includes essential files (`run.sh`, `Dockerfile`, `requirements.txt`) configured for Code Ocean's platform.
-   **Neural Network Example**: A PyTorch-based example demonstrating data loading (Iris dataset), preprocessing, modular model training, performance recording, plotting metrics, and generating an analysis report.
-   **Modular Codebase**: Organized into `models`, `data_loader`, and `utils` modules for better maintainability and reusability.
-   **Docker Compose Integration**: Facilitates easy local setup and execution of the environment and analysis.
-   **Dynamic Path Handling**: Uses `CODEOCEAN_BASE_DIR` environment variable to adapt file paths for both local (`/app`) and Code Ocean (`/`) environments.
-   **Clear Documentation**: Comprehensive `README.md` to guide users through the process.

## Directory Structure

The boilerplate includes the following directories and files:

```
codeocean-boilerplate/
├── code/                   # Contains all your source code and scripts
│   ├── __init__.py         # Makes 'code' a Python package
│   ├── data_loader/        # Module for data loading and preprocessing
│   │   ├── __init__.py     # Makes 'data_loader' a Python package
│   │   └── data_module.py  # Contains load_iris_data function
│   ├── models/             # Module for neural network models
│   │   ├── __init__.py     # Makes 'models' a Python package
│   │   └── model.py        # Contains the Net class definition
│   ├── utils/              # Module for utility functions (plotting, reporting)
│   │   ├── __init__.py     # Makes 'utils' a Python package
│   │   └── metrics_reporter.py # Contains plot_metrics and generate_report functions
│   └── nn_example.py       # Main script to run the neural network example
├── data/                   # Stores raw or processed data files (e.g., Iris.csv)
├── docs/                   # For additional documentation, notes, or supplementary materials
├── results/                # Where all generated outputs (figures, tables, reports, trained models) should be saved
├── Dockerfile              # Defines the computational environment for your capsule (PyTorch enabled)
├── README.md               # This documentation file
├── requirements.txt        # Lists Python dependencies (PyTorch, scikit-learn, matplotlib, pandas)
├── run.sh                  # The main script executed by Code Ocean to run your analysis
└── docker-compose.yml      # Configuration for local development with Docker Compose
```

## Getting Started

### Prerequisites

-   Docker installed (for local development).
-   A Code Ocean account.

### Customising the Boilerplate

To adapt this boilerplate for your own research project, follow these steps:

1.  **Clone this Repository**:
    ```bash
    git clone https://github.com/fanzhang916/codeocean-boilerplate.git
    cd codeocean-boilerplate
    ```

2.  **Replace Example Code and Data**:
    -   Place your analysis scripts in the `code/` directory, respecting the modular structure. For example, your models in `code/models`, data loaders in `code/data_loader`, and utilities in `code/utils`. Remember to add `__init__.py` files to any new directories you create that contain Python modules.
    -   For tabular data, place your `.csv` files in the `data/` directory (e.g., `Iris.csv`). Ensure that your scripts reference data using paths constructed with `os.path.join(os.environ.get('CODEOCEAN_BASE_DIR', '/'), 'data', 'your_file.csv')` for maximum compatibility.

3.  **Update `requirements.txt`**:
    -   List all Python packages required by your code, one per line.
    -   Note: If your Dockerfile uses a base image that already includes PyTorch (like the current `pytorch/pytorch` image), you might not need to explicitly list `torch` here unless you require a specific version not provided by the base image.

4.  **Modify `run.sh`**:
    -   Edit `run.sh` to execute your main analysis script(s). The current `run.sh` executes `code/nn_example.py`.
    -   Ensure that your scripts save all generated results, figures, tables, and trained models into the `/results` directory (or `os.path.join(os.environ.get('CODEOCEAN_BASE_DIR', '/'), 'results')` for dynamic pathing). Code Ocean expects all outputs to be in this directory for reproducibility checks.


5.  **Update `Dockerfile` (if necessary)**:
    -   The provided `Dockerfile` uses a PyTorch-enabled base image and sets a default `CODEOCEAN_BASE_DIR=/`.
    -   If your project requires a different base image (e.g., R, Julia, or a different PyTorch version) or additional system-level dependencies (e.g., `apt-get install some-package`), modify the `Dockerfile` accordingly.
    -   Remember to keep the `WORKDIR /` and the `RUN mkdir -p /results` lines.

### Local Development with Docker Compose

Docker Compose allows you to build and run your capsule locally, replicating the Code Ocean environment. This is useful for testing and debugging. The `docker-compose.yml` is configured to mount your project folders under `/app` inside the container and sets the `CODEOCEAN_BASE_DIR` environment variable to `/app`.

1.  **Build the Docker Image**:
    Navigate to the `codeocean-boilerplate` directory and run:
    ```bash
    docker compose build
    ```

2.  **Start the Container for Interactive Development**:
    To start the container and keep it running in the background for interactive development (e.g., connecting with VSCode Remote - Containers):
    ```bash
    docker compose up -d
    ```

3.  **Connect with VSCode Remote - Containers**:
    *   In VSCode, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and select "Remote-Containers: Attach to Running Container...".
    *   Choose the container named `codeocean-boilerplate-codeocean` (or similar).

4.  **Run Code Inside the Container (via VSCode Terminal)**:
    Once connected, open a new terminal in VSCode. You can now run your scripts:
    ```bash
    bash code/run.sh
    ```

### Running on Code Ocean

1. **Create a New Capsule**: On the Code Ocean platform, click "New Capsule" to create a new computational capsule.

2. **Configure Environment**: 
   - Select the appropriate environment (e.g., PyTorch 2.4.0, CUDA 12.4.0, Mambaforge 24.5.0-0, Python 3.12.4, Ubuntu 22.04)
   - Copy the dependencies list from your local `requirements.txt` file
   - Paste them into the pip configuration section in the Code Ocean environment settings

3. **Upload Project Files**: 
   - Upload the contents from your local folder to match the root directories on the capsule
   - Ensure the following structure is maintained:
     - `code/` → capsule root `/code/`
     - `data/` → capsule root `/data/`
     - `run.sh` → capsule root `/run.sh`
   - Note: Exclude `docker-compose.yml` as it's only for local use
   - The `CODEOCEAN_BASE_DIR` will default to `/` in the Code Ocean environment

4. **Set Execution Script**: 
   - Right-click on `run.sh` in the Code Ocean file browser
   - Select "Set as file to run" from the context menu
   - This designates `run.sh` as the main execution script for the capsule

5. **Execute the Reproducible Run**: 
   - Click "Commit Changes" to save your configuration
   - Click "Reproducibility Run" to execute the experiment
   - Code Ocean will automatically run the pipeline defined in `run.sh`

6. **Access Results**: 
   - After the run completes, navigate to the `results/` folder in the capsule
   - You will find all outputs including:
     - `iris_nn_model.pth` (trained model weights)
     - `training_metrics.png` (visualization plots)
     - `analysis_report.txt` (training summary)
   - Download or view these artifacts directly from the Code Ocean interface

---

## Example: Neural Network with PyTorch

The details of the provided example can be found here: [docs/README.md](docs/README.md)

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Suggestions for enhancing reproducibility features or supporting other languages/environments are welcome.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details. (Note: A `LICENSE.md` file would typically be created in a real project.)
