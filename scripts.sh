

scp -P 22814 -r Bayesian_PEFT root@76.69.193.132:/workspace/

ssh -p 50379 root@154.20.254.95 -L 8080:localhost:8080


rsync -avz -e 'ssh -p 50379' \
    /Users/marek/Projects/Bayesian_PEFT/ \
    root@154.20.254.95:/workspace/Bayesian_PEFT/ \
    --exclude '**/__pycache__' \
    --exclude '**/.ipynb_checkpoints' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude 'tmp' \
    --exclude 'data' \
    --exclude 'workspace' 


    ssh -p 37126 root@213.181.123.81 -L 8080:localhost:8080

rsync -avz -e 'ssh -p 37126' \
    /Users/marek/Projects/Bayesian_PEFT/ \
    root@213.181.123.81:/workspace/Bayesian_PEFT/ \
    --exclude '**/__pycache__' \
    --exclude '**/.ipynb_checkpoints' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude 'data' 

# Alternative solution (when scp does not work - e.q. on work laptop)
git add -A && git commit -m 'update' && git push 

# Setup 
pip install -e ".[examples]" && pip install bitsandbytes && apt install psmisc

# Run example
python ./examples/example_usage.py
