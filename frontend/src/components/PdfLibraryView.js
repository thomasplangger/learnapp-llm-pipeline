
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

const API_BASE = process.env.REACT_APP_BACKEND_URL || '';

export default function PdfLibraryView() {
  const [pdfs, setPdfs]         = useState([]);
  const [loadingList, setLoadingList] = useState(false);
  const [error, setError]       = useState('');
  const [actionLoading, setActionLoading] = useState(null);
  const [activePdfId, setActivePdfId] = useState(null);
  const navigate               = useNavigate();

  
  useEffect(() => {
    (async () => {
      setLoadingList(true);
      setError('');
      try {
        const res = await fetch(`${API_BASE}/api/pdfs`);
        if (!res.ok) throw new Error();
        setPdfs(await res.json());
      } catch {
        setError('Failed to load PDF library');
      } finally {
        setLoadingList(false);
      }
    })();
  }, []);

  const deletePdf = async pdf_id => {
    if (!window.confirm('Delete this PDF?')) return;
    setError('');
    setActionLoading(pdf_id);
    try {
      const res = await fetch(`${API_BASE}/api/pdf/${pdf_id}`, { method: 'DELETE' });
      if (res.status !== 204) throw new Error();
      setPdfs(ps => ps.filter(p => p.pdf_id !== pdf_id));
    } catch {
      setError('Could not delete PDF');
    } finally {
      setActionLoading(null);
    }
  };

  const downloadPdf = async (pdf) => {
    setError('');
    setActionLoading(pdf.pdf_id);
    try {
      const res = await fetch(`${API_BASE}/api/pdf/${pdf.pdf_id}/raw`);
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${pdf.filename || 'document'}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {
      setError('Could not download PDF');
    } finally {
      setActionLoading(null);
    }
  };

  const formatDate = (iso) => {
    if (!iso) return 'Unknown';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <button
            onClick={() => navigate(-1)}
            className="bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
          >
            ← Back
          </button>
          <h1 className="text-3xl font-bold">My PDFs</h1>
          <div />
        </div>

        {loadingList
          ? <p className="text-gray-600">Loading PDFs…</p>
          : pdfs.length === 0
            ? <p className="text-gray-600">No PDFs uploaded yet.</p>
            : (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {pdfs.map(pdf => (
                  <div
                    key={pdf.pdf_id}
                    className="bg-white p-6 rounded-lg shadow"
                  >
                    <div className="text-lg font-semibold mb-2">{pdf.filename}</div>
                    <div className="text-gray-600 mb-1">Uploaded: {formatDate(pdf.uploaded_at)}</div>
                    <div className="text-gray-600 mb-1">Pages: {pdf.page_count}</div>
                    <div className="text-gray-600 mb-4">Words: {pdf.word_count}</div>

                    <div className="flex flex-wrap gap-2 mb-4">
                      <button
                        onClick={() => setActivePdfId(id => id === pdf.pdf_id ? null : pdf.pdf_id)}
                        className="text-sm px-3 py-2 rounded border border-gray-300 bg-white hover:bg-gray-50"
                      >
                        {activePdfId === pdf.pdf_id ? 'Hide PDF' : 'Show PDF'}
                      </button>
                      <button
                        onClick={() => downloadPdf(pdf)}
                        disabled={actionLoading === pdf.pdf_id}
                        className={`text-sm px-3 py-2 rounded ${
                          actionLoading === pdf.pdf_id
                            ? 'bg-gray-400 text-white'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                      >
                        {actionLoading === pdf.pdf_id ? 'Downloading…' : 'Download'}
                      </button>
                      <button
                        onClick={() => deletePdf(pdf.pdf_id)}
                        disabled={actionLoading === pdf.pdf_id}
                        className={`text-sm px-3 py-2 rounded ${
                          actionLoading === pdf.pdf_id
                            ? 'bg-gray-400 text-white'
                            : 'bg-red-600 text-white hover:bg-red-700'
                        }`}
                      >
                        {actionLoading === pdf.pdf_id ? 'Deleting…' : 'Delete'}
                      </button>
                    </div>

                    {activePdfId === pdf.pdf_id && (
                      <div className="mb-4">
                        <iframe
                          title={`PDF Preview ${pdf.filename}`}
                          src={`${API_BASE}/api/pdf/${pdf.pdf_id}/raw`}
                          style={{ width: '100%', height: 420, border: '1px solid #e5e7eb', borderRadius: 6 }}
                        />
                      </div>
                    )}

                    <div className="text-sm text-gray-700">
                      <div className="font-semibold mb-1">Used in courses</div>
                      {Array.isArray(pdf.courses) && pdf.courses.length > 0 ? (
                        <div className="space-y-2">
                          {pdf.courses.map((c) => (
                            <div key={c.id} className="flex items-center justify-between">
                              <div className="text-gray-700">{c.title || c.id}</div>
                              <button
                                onClick={() => navigate(`/course/${c.id}`)}
                                className="text-sm px-2 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50"
                              >
                                Open
                              </button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-gray-500">Not used in any course.</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )
        }

        {error && (
          <div className="mt-6 bg-red-100 text-red-700 p-4 rounded">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
