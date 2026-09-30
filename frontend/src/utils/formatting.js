/**
 * Formatting helpers for numbers, currency, timestamps, and dataset metadata.
 */

export function formatNumber(val) {
  if (val === null || val === undefined || isNaN(val)) return '-';
  if (typeof val === 'number') {
    return new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 }).format(val);
  }
  return String(val);
}

export function formatCurrency(val, currency = 'INR') {
  if (val === null || val === undefined || isNaN(val)) return '-';
  const symbol = currency === 'INR' ? '₹' : '$';
  return `${symbol}${formatNumber(val)}`;
}

export function formatFileSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function formatTimestamp(timestamp) {
  if (!timestamp) return '';
  const date = typeof timestamp === 'number' ? new Date(timestamp * 1000) : new Date(timestamp);
  if (isNaN(date.getTime())) return '';
  
  const now = new Date();
  const diffHours = (now - date) / (1000 * 60 * 60);

  if (diffHours < 24 && now.getDate() === date.getDate()) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export function truncateText(text, maxLength = 30) {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength).trim() + '...';
}

