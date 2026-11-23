# VeriAI Frontend

React + Vite interface for the VeriAI FastAPI claim-verification backend.

## Requirements

- Node.js 18 or later
- npm
- The VeriAI backend running at `http://127.0.0.1:8000`

## Run locally (Windows)

Open a second Anaconda Prompt or terminal. Keep the backend running in the first terminal.

```cmd
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://127.0.0.1:5173).

The frontend calls:
- `GET /health` to show API availability.
- `GET /verification/status` to show whether the NLI model is loaded.
- `POST /verification/verify` to retrieve evidence and display the verdict.

The backend allows the local Vite origins `http://127.0.0.1:5173` and `http://localhost:5173`. For another trusted frontend origin, add it to the comma-separated `VERIAI_CORS_ORIGINS` environment variable before starting the backend.

Optionally copy `.env.example` to `.env` and set `VITE_API_BASE_URL` when the API is hosted somewhere other than `http://127.0.0.1:8000`.

## Production build

```cmd
npm run build
npm run preview
```

## Interface behavior

The page submits one claim at a time, displays the three FEVER verdicts, shows retrieved NLI evidence, and labels the model softmax score as uncalibrated. It does not present that score as a probability of factual correctness.
