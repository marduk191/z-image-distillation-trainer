git clone https://github.com/marduk191/z-image-distillation-trainer.git

cd z-image-distillation-trainer

python -m venv venv

call "G:\z-image-distillation-trainer\venv\scripts\activate"

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install git+https://github.com/huggingface/diffusers

pip install -r requirements.txt
