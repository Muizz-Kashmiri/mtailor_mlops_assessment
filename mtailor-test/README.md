# Cerebrium Image Classification Model Deployment

## Setup
Python version==3.11.12

## LOOM LINK:
[Loom](https://www.loom.com/share/ac6a278a2a3f43908094e7b5cf08b184?sid=04ff0068-0c51-441d-be45-c7560b2ef6e5)

```bash
git clone https://github.com/Muizz-Kashmiri/mtailor_mlops_assessment.git
cd MTAILOR_MLOPS_ASSESSMENT/mtailor-test
pip install -r requirements.txt
```
# Run your Flask app
python app.py

# Test the endpoint: 
python test_server.py test_images/S0.jpeg --url http://localhost:8000 # should return 0