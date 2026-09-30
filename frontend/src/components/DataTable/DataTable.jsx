import React from 'react';
import { formatNumber } from '../../utils/formatting.js';

export function DataTable({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) return null;

  // Extract column headers excluding internal keys (e.g. _internal_row_id)
  const columns = Object.keys(data[0]).filter((col) => !col.startsWith('_'));

  return (
    <div className="w-full my-3 overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-sm">
      <div className="overflow-x-auto max-h-72">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="sticky top-0 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-700">
            <tr>
              <th className="px-3.5 py-2.5 w-12 text-slate-400 font-normal border-r border-slate-200/60 dark:border-slate-700/60">
                #
              </th>
              {columns.map((col) => (
                <th key={col} className="px-3.5 py-2.5 font-semibold capitalize whitespace-nowrap">
                  {col.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 text-slate-700 dark:text-slate-300">
            {data.map((row, idx) => (
              <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                <td className="px-3.5 py-2 text-slate-400 font-mono text-[11px] border-r border-slate-100 dark:border-slate-800/60">
                  {idx + 1}
                </td>
                {columns.map((col) => {
                  const val = row[col];
                  const isNumber = typeof val === 'number';
                  return (
                    <td
                      key={col}
                      className={`px-3.5 py-2 whitespace-nowrap ${isNumber ? 'font-mono text-right' : ''}`}
                    >
                      {isNumber ? formatNumber(val) : String(val ?? '-')}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-3.5 py-1.5 bg-slate-50 dark:bg-slate-800/50 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-400 flex justify-between items-center">
        <span>Showing {data.length} records</span>
        <span>Grounded dataset query</span>
      </div>
    </div>
  );
}

