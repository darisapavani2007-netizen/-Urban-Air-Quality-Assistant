# Urban Air Quality Assistant
**NASA Space Apps Challenge 2025**

An intelligent, modular Retrieval-Augmented Generation (RAG) assistant for real-time urban air quality monitoring, atmospheric inversion analysis, and WHO-grounded health advisories.

---

## 🌟 Key Features

- **Conversational RAG Pipeline**: Combines live sensor telemetry with authoritative health guidance (WHO 2021 Global Air Quality Guidelines & EPA AQI standards).
- **Multi-Source Environmental Telemetry**: Ingests real-time air pollutant data (PM2.5, PM10, O3, NO2) and meteorological factors (temperature, wind dispersion, and boundary layer thermal inversions).
- **Personalized Health Advisories**: Dynamically tailors health recommendations for vulnerable populations—including individuals with asthma, children, seniors, outdoor athletes, and pregnant women.
- **Modern Interactive Dashboard**: Sleek dark-mode interface with a black background and light-yellowish/golden amber telemetry gauges, live chat, and a vector retrieval inspector.
- **Dual-Mode Resilient Architecture**: Operates with Google Gemini 2.5 Flash when API keys are configured, with automatic fallback to high-performance local semantic synthesis.
- **FastAPI Backend**: Modular, asynchronous REST API with automatic Swagger documentation at `/docs`.

---

## 🏗️ Project Architecture

```
Urban-Air-Quality-Assistant/
├── main.py                     # FastAPI application entrypoint & static route
├── index.html                  # Interactive web dashboard (Black & Amber theme)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git exclusions
├── README.md                   # Documentation
├── src/
│   ├── rag_pipeline.py         # Dual-mode RAG orchestrator & synthesis engine
│   ├── vector_store.py         # Vector store with cosine similarity & FAISS support
│   ├── response_formatter.py   # Type-safe health recommendation engine
│   └── data_clients/
│       ├── base_client.py      # Abstract data client interface
│       ├── openaq_client.py    # OpenAQ real-time pollutant telemetry
│       ├── weather_client.py   # Weather and atmospheric inversion client
│       ├── guidelines_client.py# WHO 2021 & EPA air quality standards
│       └── airnow_client.py    # AirNow API integration
└── data/                       # Vector store persistence (auto-generated)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Recommended: Git

### 1. Clone the Repository
```bash
git clone https://github.com/darisapavani2007-netizen/-Urban-Air-Quality-Assistant.git
cd -Urban-Air-Quality-Assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)
Copy `.env.example` to `.env` and add your API keys:
```env
GEMINI_API_KEY=your_google_gemini_api_key
OPENAQ_API_KEY=optional_openaq_key
WEATHER_API_KEY=optional_weather_key
```
*(Note: If no API keys are provided, the system automatically runs in standalone resilient RAG mode with full offline intelligence.)*

### 4. Run the Application
```bash
python main.py
```
Or with Uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Web Interface & API Docs
- **Interactive Web App**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Urban Air Quality Assistant web dashboard |
| `GET` | `/health` | Service health status (vector store, data clients, pipeline) |
| `POST` | `/chat` | Main RAG chat endpoint (`message`, `location`, `user_context`) |
| `GET` | `/data/sources` | Lists registered environmental data sources |
| `GET` | `/data/refresh` | Triggers knowledge base refresh in the background |
| `GET` | `/info` | General API and project metadata |

---

## 🛡️ Health & Vulnerability Profiles

The assistant tailors guidance according to WHO 2021 benchmarks:
- **General Public**: Safe outdoor exercise windows, ventilation practices.
- **Asthma / Respiratory**: Bronchoconstriction warnings, rescue inhaler reminders, peak flow monitoring.
- **Children**: School recess guidance, indoor air filtration thresholds.
- **Elderly / Cardiovascular**: Exertion reduction, systemic vascular stress precautions.
- **Athletes / Outdoor Workers**: Aerobic particulate intake warnings, N95/FFP2 protective gear recommendations.

---

## 📜 License

MIT License. Developed for the **NASA Space Apps Challenge 2025**.
