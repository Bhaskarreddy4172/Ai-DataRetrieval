import { useState, useCallback, useEffect } from 'react';
import { datasetApi } from '../api/datasetApi.js';

export function useDataset() {
  const [activeDataset, setActiveDataset] = useState('employees.xlsx');
  const [rowCount, setRowCount] = useState(80);
  const [columnCount, setColumnCount] = useState(8);
  const [childCount, setChildCount] = useState(0);
  const [columns, setColumns] = useState([]);
  const [columnTypes, setColumnTypes] = useState({});
  const [suggestions, setSuggestions] = useState([]);
  const [summary, setSummary] = useState('');
  const [samples, setSamples] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load dataset profile
  const fetchProfile = useCallback(async () => {
    try {
      const data = await datasetApi.getProfile();
      if (data.dataset_name) {
        setActiveDataset(data.dataset_name);
        setRowCount(data.rows || data.row_count || 0);
        if (data.child_datasets_count !== undefined) {
          setChildCount(data.child_datasets_count);
        }
        const colList = data.columns || [];
        setColumns(colList.map(c => typeof c === 'string' ? c : c.name || c.column_name));
        setColumnCount((data.columns || []).length);
        setColumnTypes(data.column_types || {});
      }
    } catch (err) {
      console.warn('Could not fetch dataset profile:', err.message);
    }
  }, []);

  // Load dataset suggestions
  const fetchSuggestions = useCallback(async () => {
    try {
      const data = await datasetApi.getSuggestions();
      if (data.suggestions && data.suggestions.length > 0) {
        setSuggestions(data.suggestions);
      }
    } catch (err) {
      console.warn('Could not fetch suggestions:', err.message);
    }
  }, []);

  // Load available sample datasets
  const fetchSamples = useCallback(async () => {
    try {
      const data = await datasetApi.getSamples();
      if (Array.isArray(data)) {
        setSamples(data);
      }
    } catch (err) {
      console.warn('Could not fetch samples:', err.message);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchProfile();
    fetchSuggestions();
    fetchSamples();
  }, [fetchProfile, fetchSuggestions, fetchSamples]);

  // Upload file
  const uploadFile = async (file) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await datasetApi.uploadDataset(file);
      if (res.status === 'success') {
        setActiveDataset(res.dataset_name);
        setRowCount(res.rows);
        setColumnCount(res.columns ? res.columns.length : 0);
        setColumns(res.columns || []);
        if (res.suggestions) setSuggestions(res.suggestions);
        if (res.summary) setSummary(res.summary);
        return { success: true, message: res.message };
      }
      throw new Error(res.message || 'Upload failed.');
    } catch (err) {
      setError(err.message);
      return { success: false, error: err.message };
    } finally {
      setIsLoading(false);
    }
  };

  // Switch sample
  const switchSample = async (sampleId) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await datasetApi.switchSample(sampleId);
      if (res.status === 'success') {
        setActiveDataset(res.active_dataset);
        setRowCount(res.rows);
        setColumnCount(res.columns ? res.columns.length : 0);
        setColumns(res.columns || []);
        if (res.suggestions) setSuggestions(res.suggestions);
        if (res.summary) setSummary(res.summary);
        return { success: true };
      }
      throw new Error(res.message || 'Switching dataset failed.');
    } catch (err) {
      setError(err.message);
      return { success: false, error: err.message };
    } finally {
      setIsLoading(false);
    }
  };

  return {
    activeDataset,
    rowCount,
    columnCount,
    childCount,
    columns,
    columnTypes,
    suggestions,
    summary,
    samples,
    isLoading,
    error,
    uploadFile,
    switchSample,
    refreshProfile: fetchProfile,
  };
}

