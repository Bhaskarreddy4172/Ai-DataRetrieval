import React, { useState } from 'react';
import { useApp } from '../../context/AppContext.jsx';
import { Modal } from '../Common/Modal.jsx';
import { Button } from '../Common/Button.jsx';
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { validateFileUpload } from '../../utils/validation.js';

export function UploadModal() {
  const { isUploadModalOpen, closeUploadModal, uploadFile } = useApp();
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleReset = () => {
    setSelectedFile(null);
    setError(null);
    setSuccess(null);
    setIsUploading(false);
  };

  const handleClose = () => {
    handleReset();
    closeUploadModal();
  };

  const handleFileSelect = (file) => {
    setError(null);
    setSuccess(null);
    const validation = validateFileUpload(file);
    if (!validation.valid) {
      setError(validation.error);
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setError(null);

    const res = await uploadFile(selectedFile);
    setIsUploading(false);

    if (res.success) {
      setSuccess(res.message || 'Dataset uploaded successfully!');
      setTimeout(() => {
        handleClose();
      }, 1500);
    } else {
      setError(res.error || 'Failed to upload dataset.');
    }
  };

  return (
    <Modal isOpen={isUploadModalOpen} onClose={handleClose} title="Upload Dataset">
      <div className="space-y-4">
        {/* Dropzone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('modal-file-input')?.click()}
          className={`p-6 border-2 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-all ${
            isDragging
              ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/30'
              : selectedFile
              ? 'border-emerald-500/50 bg-emerald-50/30 dark:bg-emerald-950/20'
              : 'border-slate-300 dark:border-slate-700 hover:border-blue-400 hover:bg-slate-50 dark:hover:bg-slate-800/50'
          }`}
        >
          <input
            id="modal-file-input"
            type="file"
            accept=".csv,.xlsx,.xls,.json"
            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
            className="hidden"
          />

          <UploadCloud className={`w-10 h-10 mb-2 ${selectedFile ? 'text-emerald-500' : 'text-slate-400'}`} />

          {selectedFile ? (
            <div className="text-center space-y-1">
              <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-800 dark:text-slate-200">
                <FileText className="w-4 h-4 text-emerald-500" />
                <span>{selectedFile.name}</span>
              </div>
              <p className="text-xs text-slate-500">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div className="text-center space-y-1">
              <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Click to browse or drag & drop file
              </p>
              <p className="text-xs text-slate-400">
                Supports CSV, XLSX, XLS, JSON (Max 50MB)
              </p>
            </div>
          )}
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-900/50 flex items-center gap-2 text-xs text-red-600 dark:text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Success Alert */}
        {success && (
          <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-900/50 flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>{success}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-2 pt-2">
          <Button variant="ghost" onClick={handleClose} disabled={isUploading}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!selectedFile || isUploading}
            icon={isUploading ? Loader2 : null}
          >
            {isUploading ? 'Ingesting Dataset...' : 'Upload & Set Active'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

