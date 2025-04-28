download Python 3.9 and add to the system environment variables path (e.g., during installation)

open powershell with administrator

in powershell navigate to the project folder (cd your_project) and create an environment, with:

    python3.9 -m venv env39

activate the environment with:

    .\env39\Scripts\activate

open your code editor (e.g., VSCode, Cursor)

open or restart the terminal in VSCode (or other IDE)

clone this project repository

navigate to the project folder in terminal

install the packages:

    pip install captum memory_profiler opencv-python requests tensorflow ipykernel pandas scikit-image scikit-learn seaborn

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

--
(for using ImageNet dataset): get huggingface access token from https://huggingface.co/settings/tokens and add in  
    
    datasets\imagenet_download.py

download the datasets:

    python datasets\download.py
--

then you are good to go. open one of the benchmark jupyter files (suggested: benchmark.ipynb)


-----------

now at that point (or also before the lines above (above probably should be general intro to explainable AI, and some words on efficiency, and how this project simplifies by providing ready benchmark) we should give proper information on the project and its capabilities) (please structure and reprase everything and design to look very nice)
* what is where: which benchmark notebooks does what. main and suggested: is benchmark.ipynb (where user selects relevant hardware and which method, model and dataset combination he wants to run), for LIME (e.g., other perturbation based models if properly extended) LIME_benchmark.ipynb (where user can select method parameters such as number of samples), local_benchmark.ipynb (easier / more cell separated version of benchmark.ipynb, but could run one combination at a time (e.g., LIME for ResNet on ImageNet sample)), and extended_benchmark.ipynb (that supports global xai methods as well, though by only aggregated heatmaps)
* how user can add his own XAI methods by creating python files under proper folder under xai_methods folder. they should follow the pattern of function names present in other py files
* how user could add ML models by adding in model zoo under models\model_loader.py or adjust code to include their own models right from there. add labels for models under models\label_utils
* currently user can select from either existing datasets (ImageNet, CIFAR-10, STL-10) or via URL to run image to get explanations on classification. in order to extend or add other capabilities in (datasets\data_loader & datasets\dataset_loader)

additionally:
* files under utils\ folder serves as utility files: checking current hardware (e.g., Nvidia..). for adding GPU/CPU models for benchmark, functions for memory measurements and energy calculations, function for loading xai method (load_xai.py), and saving results of experiments in their respective JSON and CSV files.
* all experiment results (per image efficiency results (for deeper analysis), example attribution visualizations and summary.jsonl (main result log of mean values)  as well as figures\ and tables\ generated after running experiment_results\evaluation.ipynb for EDA).
* results\ folder (generated after running evaluation.ipynb) holding figures for XAI method efficiency comparisons.


