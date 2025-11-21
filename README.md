Description: 

    FastAPI API for 3-class Nepali sentiment analysis (NEGATIVE, NEUTRAL, POSITIVE) 
    using a BERT model loaded directly from Hugging Face. No large model files are included.

setup:

  clone:
  
    - git clone https://github.com/Bijayaaaa/AI_Internship.git
    - cd AI_Internship
    - git checkout Bert
  
  virtual_environment:
    
    windows:
      - python -m venv venv
      - venv\Scripts\activate
    linux_mac:
      - python3 -m venv venv
      - source venv/bin/activate
  
  
  install_dependencies: pip install -r requirements.txt

run_local:
  
  command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  swagger_ui: http://127.0.0.1:8000/docs

example_request:
 
  endpoint: POST /predict
  
  input:
    text: "यो चलचित्र निकै राम्रो थियो"
  
  response:
    input: "यो चलचित्र निकै राम्रो थियो"
  
    prediction:
      label: POSITIVE
      probabilities:
        negative: 0.01
        neutral: 0.05
        positive: 0.94

project_structure:
  - main.py
  - requirements.txt
  - Dockerfile
  - Nepali_sentiment_analysis.ipynb
  - README.md
  - .gitignore

note: Model downloads automatically from Hugging Face; internet required on first run.

docker:
 
  build: docker build -t nepali-bert-api .
  
  run: docker run -p 8000:8000 nepali-bert-api
  
  access: http://localhost:8000/docs

