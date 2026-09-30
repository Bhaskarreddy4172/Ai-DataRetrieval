# API Documentation

The Universal Dataset Chatbot exposes REST API endpoints through FastAPI.

## Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload customer dataset (.csv, .xlsx, .json), profile, and index |
| `GET` | `/dataset/profile` | Retrieve comprehensive schema intelligence and column metadata |
| `POST` | `/chat` | Natural language question answering with 0% hallucination |
| `POST` | `/clear-chat` | Clear active conversation memory for a session |
| `POST` | `/new-session` | Initialize a new isolated session UUID |
| `GET` | `/history` | Retrieve conversation turn history |
| `GET` | `/download-chat` | Export chat history as JSON or plain text |
| `GET` | `/health` | Server and Ollama connectivity status |

---

### 1. `POST /chat`
**Request:**
```json
{
  "question": "Who has the highest salary in Engineering?",
  "session_id": "default"
}
```

**Response:**
```json
{
  "question": "Who has the highest salary in Engineering?",
  "operation": "MAX",
  "conditions": [
    {
      "column": "Department",
      "operator": "EQUALS",
      "value": "Engineering"
    }
  ],
  "results": [
    {
      "Employee ID": "EMP042",
      "Employee Name": "Priya Sharma",
      "Department": "Engineering",
      "Salary": 125000
    }
  ],
  "result_count": 1,
  "answer": "Priya Sharma has the highest salary in Engineering with ₹125,000.",
  "grounded": true,
  "confidence": 0.98,
  "processing_time": 0.12
}
```

---

### 2. `POST /upload`
**Request:** Multipart form data containing `file` (`.csv`, `.xlsx`, or `.json`).

**Response:**
```json
{
  "status": "success",
  "message": "Dataset 'customers.xlsx' successfully uploaded and active.",
  "dataset_name": "customers.xlsx",
  "rows": 1500,
  "columns": ["Customer ID", "Name", "Total Spend", "City"],
  "column_types": {
    "Customer ID": "text",
    "Name": "text",
    "Total Spend": "numeric",
    "City": "text"
  }
}
```

---

### 3. `GET /download-chat`
**Query Parameters:**
- `session_id` (string, optional, default: `"default"`)
- `format` (string, optional: `"json"` or `"txt"`)

**Response:** Attachment file download (`chat_session.json` or `chat_session.txt`).

