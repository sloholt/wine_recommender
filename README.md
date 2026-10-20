# wine_recommender

To run: 
`set "PYTHONPATH=%CD%\src"`
`python -m wine_recommender.app --csv data\raw\wines.csv`

Project Structure:
wine_app/
├── README.md
├── requirements.txt
├── .env.example              # API keys (Google, LLM, etc.)
├── .gitignore
│
├── wine_app/                 # the importable package
│   ├── __init__.py
│   ├── config.py             # 1. Setup: env vars, constants, API clients
│   ├── image_capture.py      # 2. Capture and Clean Image
│   ├── ocr.py                # 3. Google Cloud Vision OCR
│   ├── extraction.py         # 4. LLM Extraction and Validated JSON
│   ├── schemas.py            #    (Pydantic/JSON models from section 4)
│   ├── wine_data.py          # 5. Data collection, identity matching, prices
│   ├── reviews.py            # 6. Review analytics and wine profiles
│   ├── recommendations.py    # 7. Similar bottle recommendations
│   └── pipeline.py           # glue: runs 2 → 7 in notebook order
│
├── app/
│   └── main.py               # 8. App and integration (Streamlit/FastAPI/etc.)
│
├── notebooks/
│   └── original.ipynb        # keep the original for reference
├── data/                     # sample images, cached results
└── tests/
    └── test_extraction.py