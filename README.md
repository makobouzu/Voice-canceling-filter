# Voice-canceling-filter
The output is an video with a human voice and figure removed in response to video input.

## PREPARE  

python version: 3.7.7
```
python -m venv venv
source venv/bin/activate.fish

pip install -r requirements.txt
```

### Files  
input/  
-- sample.mp4  

output/

## RUN  
```
python run.py -input_file ./input/sample.mp4 -output_dir ./output
```
