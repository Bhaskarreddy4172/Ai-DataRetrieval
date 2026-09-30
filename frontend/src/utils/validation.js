/**
 * Validation utilities for dataset uploads and input questions.
 */

export const ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.json'];
export const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

export function validateFileUpload(file) {
  if (!file) {
    return { valid: false, error: 'No file selected.' };
  }

  const name = file.name.toLowerCase();
  const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext));

  if (!hasValidExt) {
    return {
      valid: false,
      error: `Invalid file format. Allowed formats: ${ALLOWED_EXTENSIONS.join(', ')}`,
    };
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      valid: false,
      error: `File size exceeds 50MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB).`,
    };
  }

  return { valid: true, error: null };
}

export function sanitizeQuestion(q) {
  if (!q) return '';
  return q.trim().replace(/\s+/g, ' ');
}

