# Text and Image Guided 3D Avatar Generation and Manipulation

## Installation and Usage

1. Install the dependencies in env.yml
    
    `conda env create -f env.yml`

    `conda activate latent3d-env`
    
2. Install pytorch3d by running the following:

    ```python
    import sys
    import torch
    import subprocess
    version_str="".join([
        f"py3{sys.version_info.minor}_cu",
        torch.version.cuda.replace(".",""),
        f"_pyt{torch.__version__.split("+")[0].replace(".", "")}"
    ])
    subprocess.run(f"conda install https://anaconda.org/pytorch3d/pytorch3d/0.6.2/download/linux-64/pytorch3d-0.6.2-{version_str}.tar.bz2", shell=True)
    ```

3. Download the pre-trained TBGAN model from this link: https://ibug.doc.ic.ac.uk/resources/tbgan/. 

* Note: You will be required to fill the form for 'End User Licence Agreement'.

4. Place the pre-trained TBGAN model snapshot in the '/models/snapshots' directory. 

5. Download the pre-trained ArcFace model

    `wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O model_ir_se50.pth`

6. Place the pre-trained ArcFace model in the '/models/snapshots' directory. 

### Text-Based Manipulations

Running the 'optimize.py' with the required parameters and list of text prompts for manipulation, the rendered manipulated 3D faces and their originals will be saved under the direcory './results'. 

`python3 optimize.py --seed [SEED] --mode [MODE] --text_list [TEXT_LIST] --num_epochs [NUM_EPOCHS] --lambda_id [ID_COEFF] --lambda_l2 [L2_COEFF] --learning_rate [LEARNING_RATE]`

* Example: `python3 optimize.py --seed 1000 --mode text-based --text_list "happy human" "happy person" --num_epochs 100 --lambda_id 0.01 --lambda_l2 0.0 --learning_rate 0.01 --folder_title happy`

### Image-based Manipulations

Running the 'optimize.py' with given parameters and an image for manipulation, the rendered manipulated 3D faces and their originals will be saved under the direcory './results'. 

`python3 optimize.py --seed [SEED] --mode [MODE] --image_path [IMAGE_PATH] --num_epochs [NUM_EPOCHS] --lambda_id [ID_COEFF] --lambda_l2 [L2_COEFF] --learning_rate [LEARNING_RATE]`

### Generate Mesh from Intermediate Latent Vector

`python3 sample.py --functionality latent_vector --result_subdir ./results/003-optimize`
