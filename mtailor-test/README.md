# Cerebrium Image Classification Model Deployment

## Setup
Python version==3.11.12

```bash
git clone https://github.com/Muizz-Kashmiri/mtailor_mlops_assessment.git
cd MTAILOR_MLOPS_ASSESSMENT
pip install -r requirements.txt
```
# Run your Flask app
python app.py

# Test the endpoint: 
python test_server.py test_images/S0.jpeg --url http://localhost:8000 # should return 0