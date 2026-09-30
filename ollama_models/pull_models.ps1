# PowerShell helper script to download and configure Ollama models for the Universal Dataset Chatbot
Write-Host "Pulling supported Ollama models for Universal Dataset Chatbot..." -ForegroundColor Cyan

$models = @("llama3.1:8b", "mistral:7b", "qwen2.5:7b", "gemma3:7b")

foreach ($m in $models) {
    Write-Host "Checking / Pulling $m..." -ForegroundColor Yellow
    ollama pull $m
}

Write-Host "All supported models pulled successfully!" -ForegroundColor Green
Write-Host "Create custom modelfiles with:" -ForegroundColor Cyan
Write-Host "  ollama create dataset-llama3 -f Modelfile.llama3"
Write-Host "  ollama create dataset-mistral -f Modelfile.mistral"
Write-Host "  ollama create dataset-qwen -f Modelfile.qwen2.5"
Write-Host "  ollama create dataset-gemma -f Modelfile.gemma3"

