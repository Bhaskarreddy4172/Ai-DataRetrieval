# REST API Specification

## Base URL
Default local endpoint: `http://127.0.0.1:8000`

---

## Endpoint Definitions

### 1. Dataset Upload
- **URL**: `/dataset/upload` (or `/upload`)
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file` (File)
- **Supported Formats**: `.csv`, `.xlsx`, `.xls`, `.json`
- **Response**:
```json
{
  "status": "success",
  "message": "Dataset 'sample_employees.xlsx' successfully uploaded and active.",
  "dataset_name": "sample_employees.xlsx",
  "rows": 100,
  "columns": ["Employee ID", "Employee Name", "Department", "Designation", "City", "Salary"],
  "sheets": ["Employees"],
  "summary": "This dataset contains 100 employee records across 6 columns..."
}
```

---

### 2. Process Natural Language Query
- **URL**: `/chat` (or `/query`)
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "question": "Who has the highest salary in Hyderabad?",
  "session_id": "default"
}
```
- **Response**:
```json
{
  "question": "Who has the highest salary in Hyderabad?",
  "operation": "MAX",
  "conditions": [{"column": "City", "operator": "=", "value": "Hyderabad"}],
  "results": [
    {
      "Employee ID": "EMP042",
      "Employee Name": "Rahul Sharma",
      "Department": "Engineering",
      "City": "Hyderabad",
      "Salary": 95000
    }
  ],
  "result_count": 1,
  "answer": "Rahul Sharma has the highest salary in Hyderabad with ₹95,000.",
  "dataset_name": "sample_employees.xlsx",
  "processing_time": 0.0452,
  "grounded": true,
  "source_type": "DATASET"
}
```

---

### 3. System Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
```json
{
  "status": "healthy",
  "active_dataset": "sample_employees.xlsx",
  "rows": 100,
  "columns": ["Employee ID", "Employee Name", "Department", "Designation", "City", "Salary"],
  "sheets": ["Employees"],
  "active_sheet": "Employees",
  "ollama": "online",
  "ollama_status": "READY",
  "ollama_details": {
    "status": "READY",
    "available": true,
    "selected_model": "llama3.1:8b"
  }
}
```

---

### 4. Conversation Management
- **`POST /conversation/clear`** (or `/clear-chat`): Clear dialogue state for a session.
- **`POST /new-session`**: Generate a new isolated session UUID.
- **`GET /history?session_id=default`**: Retrieve conversation turns.
- **`GET /download-chat?session_id=default&format=json`**: Export chat history (JSON or TXT).

---

### 5. Dataset Profile & Analytics
- **`GET /dataset/profile`**: Returns column data types, missing values, entities, and candidate keys.
- **`GET /dataset/schema`**: Compact LLM-friendly schema representation.
- **`GET /dataset/suggestions`**: Returns dynamically generated query suggestions.
- **`POST /dataset/compare`**: Performs side-by-side entity comparison.

