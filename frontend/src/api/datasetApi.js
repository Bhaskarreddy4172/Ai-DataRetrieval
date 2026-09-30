import { request } from './client.js';

export const datasetApi = {
  /**
   * Upload client dataset file (CSV, XLSX, XLS, JSON).
   */
  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);

    return request('/upload', {
      method: 'POST',
      body: formData,
    });
  },

  /**
   * List preloaded sample datasets.
   */
  async getSamples() {
    return request('/dataset/samples', {
      method: 'GET',
    });
  },

  /**
   * Switch active dataset to a preloaded sample.
   */
  async switchSample(sample_id) {
    return request(`/dataset/sample/${encodeURIComponent(sample_id)}`, {
      method: 'POST',
    });
  },

  /**
   * Get active dataset statistical profile & columns.
   */
  async getProfile() {
    return request('/dataset/profile', {
      method: 'GET',
    });
  },

  /**
   * Get dataset-agnostic suggested questions.
   */
  async getSuggestions() {
    return request('/dataset/suggestions', {
      method: 'GET',
    });
  },

  /**
   * Get natural language summary of active dataset.
   */
  async getSummary() {
    return request('/dataset/summary', {
      method: 'GET',
    });
  },

  /**
   * Switch active sheet in Excel workbook.
   */
  async switchSheet(sheet_name) {
    return request(`/dataset/sheet/${encodeURIComponent(sheet_name)}`, {
      method: 'POST',
    });
  },
};

