

cd /workspace/Bayesian_PEFT
rm -rf venv 
python3 -m venv venv
source venv/bin/activate

pip install -U pip
pip install -e ".[examples]" && pip install bitsandbytes && apt install psmisc

